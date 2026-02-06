"""HTTP retrieval service for image names using shared dense embeddings."""
from __future__ import annotations

import argparse
import json
import logging
from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import faiss
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModel, AutoTokenizer

from search_engine.servers.embedding_client import RemoteEmbeddingClient
from search_engine.servers.vector_search_client import VectorSearchClient


logger = logging.getLogger(__name__)


def load_corpus(corpus_path: Path) -> List[dict]:
    records: List[dict] = []
    with corpus_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no}: {exc}") from exc
            record.setdefault("id", line_no - 1)
            record.setdefault("name", str(record.get("filename", "")))
            record.setdefault("filename", record.get("filename", record["name"]))
            if "image_path" not in record:
                raise ValueError(f"Record on line {line_no} missing 'image_path'")
            aliases = record.get("aliases")
            if aliases is not None and not isinstance(aliases, (list, tuple)):
                raise ValueError(f"Record on line {line_no} has invalid 'aliases' field (expected list)")
            records.append(record)
    if not records:
        raise ValueError("Corpus is empty")
    return records


def _select_device(device: str) -> torch.device:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available; falling back to CPU")
        device = "cpu"
    return torch.device(device)


def _pool_embeddings(model_output, attention_mask, pooling: str = "mean") -> torch.Tensor:
    if pooling == "mean":
        last_hidden = model_output.last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
        masked = last_hidden * mask
        summed = masked.sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-6)
        return summed / denom
    if pooling == "cls":
        return model_output.last_hidden_state[:, 0]
    if pooling == "pooler" and getattr(model_output, "pooler_output", None) is not None:
        return model_output.pooler_output
    raise ValueError(f"Unsupported pooling method: {pooling}")


class EmbeddingBackend:
    def encode(self, texts: Sequence[str], *, is_query: bool) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError


class LocalEmbeddingBackend(EmbeddingBackend):
    def __init__(
        self,
        model_name: str,
        device: torch.device,
        max_length: int,
        pooling_method: str,
        use_fp16: bool,
        query_prefix: str,
        passage_prefix: str,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.pooling_method = pooling_method
        self.use_fp16 = use_fp16 and device.type == "cuda"
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix

        logger.info("Loading encoder model %s on %s", model_name, device)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.model.to(device)
        if self.use_fp16:
            self.model = self.model.half()
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)

    @torch.no_grad()
    def encode(self, texts: Sequence[str], *, is_query: bool) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        prepared = self._prepare_inputs(list(texts), is_query=is_query)
        inputs = self.tokenizer(
            prepared,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs, return_dict=True)
        embeddings = _pool_embeddings(outputs, inputs["attention_mask"], pooling=self.pooling_method)
        embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
        return embeddings.detach().cpu().to(dtype=torch.float32).numpy()

    def _prepare_inputs(self, texts: List[str], *, is_query: bool) -> List[str]:
        if "e5" in self.model_name.lower():
            prefix = self.query_prefix if is_query else self.passage_prefix
            return [f"{prefix}{text}" for text in texts]
        if "bge" in self.model_name.lower() and is_query:
            return [
                f"Represent this sentence for searching relevant passages: {text}" for text in texts
            ]
        return texts


class RemoteEmbeddingBackend(EmbeddingBackend):
    def __init__(self, endpoint: str, timeout: float, max_batch_size: int) -> None:
        self.client = RemoteEmbeddingClient(endpoint, timeout=timeout, max_batch_size=max_batch_size)

    def encode(self, texts: Sequence[str], *, is_query: bool) -> np.ndarray:
        return self.client.encode(texts, is_query=is_query)


@dataclass(frozen=True)
class IndexedEntry:
    record: Dict
    alias: str


class ImageNameIndex:
    def __init__(self, records: List[dict], backend: EmbeddingBackend, batch_size: int = 128) -> None:
        self.backend = backend
        self.batch_size = batch_size

        texts: List[str] = []
        entries: List[IndexedEntry] = []
        for record in records:
            base = str(record.get("name") or record.get("filename") or "").strip()
            if base:
                texts.append(base)
                entries.append(IndexedEntry(record=record, alias=base))

            for alias in record.get("aliases", []) or []:
                alias_str = str(alias).strip()
                if alias_str:
                    texts.append(alias_str)
                    entries.append(IndexedEntry(record=record, alias=alias_str))

        if not texts:
            raise ValueError("No valid names or aliases found in corpus")

        embeddings = self._encode_texts(texts)
        self.dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)
        self.entries = entries

    def _encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        chunks: List[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            chunk = texts[start : start + self.batch_size]
            embeddings = self.backend.encode(chunk, is_query=False)
            chunks.append(embeddings)
        return np.concatenate(chunks, axis=0)

    def search(self, query: str, topk: int) -> List[Tuple[Dict, float]]:
        if not query.strip():
            return []
        query_emb = self.backend.encode([query], is_query=True)
        scores, idxs = self.index.search(query_emb, k=min(topk, len(self.entries)))
        scored: List[Tuple[Dict, float]] = []
        seen_ids: "OrderedDict[int, Tuple[Dict, float]]" = OrderedDict()
        for idx, score in zip(idxs[0], scores[0]):
            entry = self.entries[idx]
            record = entry.record
            rec_id = int(record.get("id", idx))
            score_val = float(score)
            if rec_id not in seen_ids:
                seen_ids[rec_id] = (record, score_val)
            else:
                _, existing_score = seen_ids[rec_id]
                if score_val > existing_score:
                    seen_ids[rec_id] = (record, score_val)
            if len(seen_ids) >= topk:
                break
        scored.extend(seen_ids.values())
        return scored[:topk]


class QueryRequest(BaseModel):
    queries: List[str]
    topk: Optional[int] = None
    return_scores: bool = True


class Config:
    def __init__(
        self,
        corpus_path: Path,
        topk: int,
        backend_type: str,
        model_name: Optional[str],
        device: Optional[torch.device],
        pooling_method: Optional[str],
        max_length: Optional[int],
        use_fp16: bool,
        embedding_endpoint: Optional[str],
        vector_client: Optional[VectorSearchClient],
        vector_collection: Optional[str],
    ) -> None:
        self.corpus_path = corpus_path
        self.topk = topk
        self.backend_type = backend_type
        self.model_name = model_name
        self.device = device
        self.pooling_method = pooling_method
        self.max_length = max_length
        self.use_fp16 = use_fp16
        self.embedding_endpoint = embedding_endpoint
        self.vector_client = vector_client
        self.vector_collection = vector_collection


app = FastAPI()
index: Optional[ImageNameIndex]
config: Config
query_backend: EmbeddingBackend


@app.post("/retrieve")
def retrieve_endpoint(request: QueryRequest):
    if not request.queries:
        raise HTTPException(status_code=400, detail="'queries' must be a non-empty list")

    topk = request.topk or config.topk
    topk = max(1, topk)

    if config.vector_client is not None:
        embeddings = query_backend.encode(request.queries, is_query=True)
        hits_batch = config.vector_client.search(embeddings, collection=config.vector_collection, topk=topk)
        results: List[List[dict]] = []
        for hits in hits_batch:
            if request.return_scores:
                formatted = [
                    {
                        "document": hit.get("document"),
                        "score": round(float(hit.get("score", 0.0)), 6),
                    }
                    for hit in hits
                ]
            else:
                formatted = [hit.get("document") for hit in hits]
            results.append(formatted)
        return {"result": results}

    if index is None:
        raise HTTPException(status_code=503, detail="Image index not initialised")

    results = []
    for query in request.queries:
        matches = index.search(query, topk)
        if request.return_scores:
            formatted = [
                {
                    "document": {
                        "name": match[0].get("name") or match[0].get("filename"),
                        "filename": match[0].get("filename"),
                        "image_path": match[0].get("image_path"),
                        "aliases": match[0].get("aliases"),
                    },
                    "score": round(match[1], 6),
                }
                for match in matches
            ]
        else:
            formatted = [
                {
                    "name": match[0].get("name") or match[0].get("filename"),
                    "filename": match[0].get("filename"),
                    "image_path": match[0].get("image_path"),
                    "aliases": match[0].get("aliases"),
                }
                for match in matches
            ]
        results.append(formatted)
    return {"result": results}


@lru_cache(maxsize=1)
def _load_corpus_cached(path: Path) -> List[dict]:
    return load_corpus(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch image name retrieval service (E5 + FAISS).")
    parser.add_argument("--corpus_path", type=Path, required=True, help="Path to image JSONL corpus.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8001, help="Server port")
    parser.add_argument("--topk", type=int, default=3, help="Default number of results to return.")
    parser.add_argument("--embedding_endpoint", type=str, default=None, help="Remote embedding service endpoint.")
    parser.add_argument("--embedding_timeout", type=float, default=30.0, help="Timeout for embedding requests (s).")
    parser.add_argument(
        "--embedding_max_batch",
        type=int,
        default=128,
        help="Max batch size per embedding request when using remote service.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="intfloat/e5-base-v2",
        help="Model name or path for local text encoder.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device selection for local encoder (auto | cpu | cuda).",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=256,
        help="Maximum sequence length for the encoder tokenizer.",
    )
    parser.add_argument(
        "--pooling_method",
        type=str,
        default="mean",
        choices=["mean", "cls", "pooler"],
        help="Pooling strategy to obtain sentence embeddings.",
    )
    parser.add_argument(
        "--use_fp16",
        action="store_true",
        help="Enable FP16 inference for the local encoder when running on GPU.",
    )
    parser.add_argument(
        "--query_prefix",
        type=str,
        default="query: ",
        help="Prefix applied to queries for E5-family models (local backend).",
    )
    parser.add_argument(
        "--passage_prefix",
        type=str,
        default="passage: ",
        help="Prefix applied to passages for E5-family models (local backend).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for corpus encoding.",
    )
    parser.add_argument(
        "--vector_endpoint",
        type=str,
        default=None,
        help="Optional vector search service endpoint (e.g., http://127.0.0.1:9200/search).",
    )
    parser.add_argument(
        "--vector_collection",
        type=str,
        default="image_names",
        help="Collection name used when querying the vector search service.",
    )
    parser.add_argument(
        "--vector_timeout",
        type=float,
        default=30.0,
        help="Timeout (seconds) for vector search requests.",
    )
    parser.add_argument("--log_level", type=str, default="info", help="uvicorn log level")
    args = parser.parse_args()

    global index, config, query_backend

    vector_client: Optional[VectorSearchClient] = None
    if args.vector_endpoint:
        vector_client = VectorSearchClient(endpoint=args.vector_endpoint, timeout=args.vector_timeout)

    corpus_records: Optional[List[dict]] = None
    if vector_client is None:
        corpus_records = _load_corpus_cached(args.corpus_path.expanduser().resolve())

    if args.embedding_endpoint:
        backend: EmbeddingBackend = RemoteEmbeddingBackend(
            endpoint=args.embedding_endpoint,
            timeout=args.embedding_timeout,
            max_batch_size=args.embedding_max_batch,
        )
        device = None
        use_fp16 = False
        backend_type = "remote"
    else:
        device = _select_device(args.device)
        use_fp16 = bool(args.use_fp16 and device.type == "cuda")
        backend = LocalEmbeddingBackend(
            model_name=args.model_name,
            device=device,
            max_length=args.max_length,
            pooling_method=args.pooling_method,
            use_fp16=use_fp16,
            query_prefix=args.query_prefix,
            passage_prefix=args.passage_prefix,
        )
        backend_type = "local"

    query_backend = backend

    if vector_client is None and corpus_records is not None:
        index = ImageNameIndex(corpus_records, backend=backend, batch_size=args.batch_size)
    else:
        index = None
    config = Config(
        corpus_path=args.corpus_path,
        topk=args.topk,
        backend_type=backend_type,
        model_name=args.model_name if backend_type == "local" else None,
        device=device,
        pooling_method=args.pooling_method if backend_type == "local" else None,
        max_length=args.max_length if backend_type == "local" else None,
        use_fp16=bool(use_fp16),
        embedding_endpoint=args.embedding_endpoint,
        vector_client=vector_client,
        vector_collection=args.vector_collection,
    )

    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()

