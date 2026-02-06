"""Shared FAISS vector search service for text and image collections."""
from __future__ import annotations

import argparse
import json
import logging
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence

import faiss
import numpy as np
import uvicorn
from datasets import load_dataset
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from .embedding_client import RemoteEmbeddingClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class SearchRequest(BaseModel):
    collection: str
    embeddings: List[List[float]]
    topk: int = Field(default=3, ge=1, le=1000)


# ---------------------------------------------------------------------------
# Collection implementations
# ---------------------------------------------------------------------------


@dataclass
class SearchHit:
    document: Dict
    score: float


class VectorCollection:
    name: str

    def search(self, embeddings: np.ndarray, topk: int) -> List[List[SearchHit]]:  # pragma: no cover - interface
        raise NotImplementedError


class FaissFileCollection(VectorCollection):
    def __init__(
        self,
        name: str,
        index_path: Path,
        corpus_path: Path,
        *,
        faiss_gpu: bool,
        dataset_split: str = "train",
    ) -> None:
        self.name = name
        self.index_path = index_path
        self.corpus_path = corpus_path
        self.faiss_gpu = faiss_gpu
        self.dataset_split = dataset_split

        if not self.index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {self.index_path}")
        if not self.corpus_path.exists():
            raise FileNotFoundError(f"Corpus file not found: {self.corpus_path}")

        logger.info("Loading FAISS index for collection '%s' from %s", self.name, self.index_path)
        self.index = faiss.read_index(str(self.index_path))
        if self.faiss_gpu:
            co = faiss.GpuMultipleClonerOptions()
            co.useFloat16 = True
            co.shard = True
            logger.info("Moving FAISS index '%s' to GPU", self.name)
            self.index = faiss.index_cpu_to_all_gpus(self.index, co=co)

        logger.info("Loading corpus for collection '%s' from %s", self.name, self.corpus_path)
        self.corpus = load_dataset("json", data_files=str(self.corpus_path), split=self.dataset_split)
        self.dimension = self.index.d

    def search(self, embeddings: np.ndarray, topk: int) -> List[List[SearchHit]]:
        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension mismatch for collection '{self.name}': "
                f"expected {self.dimension}, got {embeddings.shape[1]}"
            )
        scores, idxs = self.index.search(embeddings, k=min(topk, self.index.ntotal))
        if scores.ndim == 1:
            scores = scores[None, :]
            idxs = idxs[None, :]

        results: List[List[SearchHit]] = []
        for row_scores, row_indices in zip(scores, idxs, strict=True):
            hits: List[SearchHit] = []
            for score, idx in zip(row_scores, row_indices, strict=True):
                if idx < 0:
                    continue
                doc = self.corpus[int(idx)]
                hits.append(SearchHit(document=dict(doc), score=float(score)))
            results.append(hits)
        return results


class ImageAliasCollection(VectorCollection):
    def __init__(
        self,
        name: str,
        corpus_path: Path,
        *,
        embedding_client: RemoteEmbeddingClient,
        batch_size: int = 128,
        faiss_gpu: bool = False,
    ) -> None:
        self.name = name
        self.corpus_path = corpus_path
        self.embedding_client = embedding_client
        self.batch_size = batch_size
        self.faiss_gpu = faiss_gpu

        if not self.corpus_path.exists():
            raise FileNotFoundError(f"Image corpus not found: {self.corpus_path}")

        logger.info("Loading image corpus for collection '%s' from %s", self.name, self.corpus_path)
        self.records: List[Dict] = self._load_corpus(self.corpus_path)
        self.entries: List[Dict] = []
        texts: List[str] = []
        for record in self.records:
            base = str(record.get("name") or record.get("filename") or "").strip()
            if base:
                texts.append(base)
                self.entries.append({"record": record, "alias": base})
            for alias in record.get("aliases", []) or []:
                alias_str = str(alias).strip()
                if alias_str:
                    texts.append(alias_str)
                    self.entries.append({"record": record, "alias": alias_str})
        if not texts:
            raise ValueError(f"Collection '{self.name}' corpus contains no names or aliases")

        logger.info(
            "Encoding %d aliases for collection '%s' via embedding service %s",
            len(texts),
            self.name,
            self.embedding_client.endpoint,
        )
        embeddings = self._encode_aliases(texts)
        self.dimension = embeddings.shape[1]

        logger.info("Building FAISS index for collection '%s'", self.name)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)
        if self.faiss_gpu:
            logger.info("Moving image alias index '%s' to GPU", self.name)
            self.index = faiss.index_cpu_to_all_gpus(self.index)

    @staticmethod
    def _load_corpus(path: Path) -> List[Dict]:
        records: List[Dict] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_no, raw in enumerate(handle, start=1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    record = json.loads(raw)
                except json.JSONDecodeError as exc:  # pragma: no cover - invalid data path
                    raise ValueError(f"Invalid JSON on line {line_no}: {exc}") from exc
                if "image_path" not in record:
                    raise ValueError(f"Record on line {line_no} is missing 'image_path'")
                record.setdefault("id", len(records))
                record.setdefault("name", record.get("filename"))
                records.append(record)
        if not records:
            raise ValueError("Image corpus is empty")
        return records

    def _encode_aliases(self, texts: Sequence[str]) -> np.ndarray:
        chunks: List[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            chunk = list(texts[start : start + self.batch_size])
            embeddings = self.embedding_client.encode(chunk, is_query=False)
            chunks.append(embeddings)
        return np.concatenate(chunks, axis=0)

    def search(self, embeddings: np.ndarray, topk: int) -> List[List[SearchHit]]:
        scores, idxs = self.index.search(embeddings, k=min(topk * 5, len(self.entries)))
        results: List[List[SearchHit]] = []
        for row_scores, row_indices in zip(scores, idxs, strict=True):
            seen: "OrderedDict[int, SearchHit]" = OrderedDict()
            for score, idx in zip(row_scores, row_indices, strict=True):
                if idx < 0:
                    continue
                entry = self.entries[int(idx)]
                record = entry["record"]
                rec_id = int(record.get("id", idx))
                score_val = float(score)
                hit = SearchHit(
                    document={
                        "name": record.get("name") or record.get("filename"),
                        "filename": record.get("filename"),
                        "image_path": record.get("image_path"),
                        "aliases": record.get("aliases"),
                    },
                    score=score_val,
                )
                if rec_id not in seen or score_val > seen[rec_id].score:
                    seen[rec_id] = hit
                if len(seen) >= topk:
                    break
            results.append(list(seen.values())[:topk])
        return results


# ---------------------------------------------------------------------------
# Service wiring
# ---------------------------------------------------------------------------


class VectorSearchService:
    def __init__(self) -> None:
        self.collections: Dict[str, VectorCollection] = {}

    def register(self, collection: VectorCollection) -> None:
        if collection.name in self.collections:
            raise ValueError(f"Collection '{collection.name}' already registered")
        logger.info("Registered collection '%s'", collection.name)
        self.collections[collection.name] = collection

    def list_collections(self) -> List[str]:
        return sorted(self.collections.keys())

    def search(self, name: str, embeddings: np.ndarray, topk: int) -> List[List[SearchHit]]:
        collection = self.collections.get(name)
        if collection is None:
            raise KeyError(name)
        return collection.search(embeddings, topk)


service = VectorSearchService()
app = FastAPI()


@app.get("/collections")
def list_collections():
    return {"collections": service.list_collections()}


@app.post("/search")
def search_endpoint(request: SearchRequest):
    if not request.embeddings:
        raise HTTPException(status_code=400, detail="'embeddings' must be a non-empty list")
    try:
        embeddings = np.asarray(request.embeddings, dtype="float32")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Invalid embeddings payload: {exc}") from exc
    if embeddings.ndim != 2:
        raise HTTPException(status_code=400, detail="'embeddings' must be a 2D array")
    try:
        hits = service.search(request.collection, embeddings, request.topk)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown collection '{request.collection}'")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    response: List[List[Dict]] = []
    for row in hits:
        formatted = [
            {
                "document": hit.document,
                "score": hit.score,
            }
            for hit in row
        ]
        response.append(formatted)
    return {"result": response}


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------


class BaseCollectionSettings(BaseModel):
    name: str
    faiss_gpu: bool = False


class FaissFileSettings(BaseCollectionSettings):
    type: Literal["faiss_file"] = "faiss_file"
    index_path: str
    corpus_path: str
    dataset_split: str = "train"


class ImageAliasSettings(BaseCollectionSettings):
    type: Literal["image_alias"] = "image_alias"
    corpus_path: str
    embedding_endpoint: str
    embedding_timeout: float = 30.0
    embedding_max_batch: int = 128
    batch_size: int = 128


class ConfigFile(BaseModel):
    collections: List[BaseCollectionSettings]

    @classmethod
    def parse(cls, path: Path) -> "ConfigFile":
        with path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        collections = []
        for item in raw.get("collections", []):
            ctype = item.get("type")
            if ctype == "faiss_file":
                collections.append(FaissFileSettings(**item))
            elif ctype == "image_alias":
                collections.append(ImageAliasSettings(**item))
            else:
                raise ValueError(f"Unsupported collection type: {ctype}")
        return cls(collections=collections)


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


def _build_collections(config_path: Path) -> None:
    cfg = ConfigFile.parse(config_path)
    for settings in cfg.collections:
        if isinstance(settings, FaissFileSettings):
            collection = FaissFileCollection(
                name=settings.name,
                index_path=Path(settings.index_path).expanduser().resolve(),
                corpus_path=Path(settings.corpus_path).expanduser().resolve(),
                faiss_gpu=settings.faiss_gpu,
                dataset_split=settings.dataset_split,
            )
            service.register(collection)
        elif isinstance(settings, ImageAliasSettings):
            client = RemoteEmbeddingClient(
                endpoint=settings.embedding_endpoint,
                timeout=settings.embedding_timeout,
                max_batch_size=settings.embedding_max_batch,
            )
            collection = ImageAliasCollection(
                name=settings.name,
                corpus_path=Path(settings.corpus_path).expanduser().resolve(),
                embedding_client=client,
                batch_size=settings.batch_size,
                faiss_gpu=settings.faiss_gpu,
            )
            service.register(collection)
        else:  # pragma: no cover - defensive
            raise RuntimeError(f"Unhandled collection settings type: {type(settings)!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the shared FAISS vector search service.")
    parser.add_argument("--config", type=Path, required=True, help="Path to JSON configuration file.")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9200)
    parser.add_argument("--log_level", type=str, default="info")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    _build_collections(args.config.expanduser().resolve())

    if not service.collections:
        raise RuntimeError("No collections registered; check configuration file")

    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()

