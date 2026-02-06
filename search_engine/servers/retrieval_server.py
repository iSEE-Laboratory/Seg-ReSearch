"""Unified retrieval server supporting local or shared embedding backends."""
from __future__ import annotations

import argparse
import json
import logging
import warnings
from dataclasses import dataclass
from typing import List, Optional, Sequence

import datasets
import faiss
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from search_engine.servers.embedding_client import RemoteEmbeddingClient
from search_engine.servers.vector_search_client import VectorSearchClient


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------


def load_corpus(corpus_path: str):
    return datasets.load_dataset("json", data_files=corpus_path, split="train", num_proc=4)


def load_docs(corpus, doc_idxs):
    return [corpus[int(idx)] for idx in doc_idxs]


# ---------------------------------------------------------------------------
# Embedding backends
# ---------------------------------------------------------------------------


def _select_device(device: str) -> torch.device:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available; falling back to CPU")
        device = "cpu"
    return torch.device(device)


def _pool_embeddings(model_output, attention_mask, pooling_method: str):
    if pooling_method == "mean":
        last_hidden = model_output.last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
        masked = last_hidden * mask
        summed = masked.sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-6)
        return summed / denom
    if pooling_method == "cls":
        return model_output.last_hidden_state[:, 0]
    if pooling_method == "pooler" and getattr(model_output, "pooler_output", None) is not None:
        return model_output.pooler_output
    raise ValueError(f"Unsupported pooling method: {pooling_method}")


class EmbeddingBackend:
    def encode(self, texts: Sequence[str], *, is_query: bool) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError


class LocalEmbeddingBackend(EmbeddingBackend):
    def __init__(
        self,
        model_name: str,
        device: torch.device,
        pooling_method: str,
        max_length: int,
        use_fp16: bool,
        query_prefix: str,
        passage_prefix: str,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.pooling_method = pooling_method
        self.max_length = max_length
        self.use_fp16 = use_fp16 and device.type == "cuda"
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix

        logger.info("Loading encoder %s on %s", model_name, device)
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
        embeddings = _pool_embeddings(outputs, inputs["attention_mask"], self.pooling_method)
        if "dpr" not in self.model_name.lower():
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


# ---------------------------------------------------------------------------
# Retriever implementations
# ---------------------------------------------------------------------------


class BaseRetriever:
    def __init__(self, config):
        self.config = config
        self.retrieval_method = config.retrieval_method
        self.topk = config.retrieval_topk
        self.index_path = config.index_path
        self.corpus_path = config.corpus_path

    def _search(self, query: str, num: int, return_score: bool):  # pragma: no cover - interface
        raise NotImplementedError

    def _batch_search(self, query_list: Sequence[str], num: int, return_score: bool):  # pragma: no cover
        raise NotImplementedError

    def search(self, query: str, num: Optional[int] = None, return_score: bool = True):
        return self._search(query, num, return_score)

    def batch_search(self, query_list: Sequence[str], num: Optional[int] = None, return_score: bool = True):
        return self._batch_search(query_list, num, return_score)


class BM25Retriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        from pyserini.search.lucene import LuceneSearcher

        self.searcher = LuceneSearcher(self.index_path)
        self.contain_doc = self._check_contain_doc()
        if not self.contain_doc:
            self.corpus = load_corpus(self.corpus_path)
        self.max_process_num = 8

    def _check_contain_doc(self):
        return self.searcher.doc(0).raw() is not None

    def _search(self, query: str, num: int = None, return_score: bool = True):
        if num is None:
            num = self.topk
        hits = self.searcher.search(query, num)
        if len(hits) < 1:
            if return_score:
                return [], []
            else:
                return []
        scores = [hit.score for hit in hits]
        if len(hits) < num:
            warnings.warn("Not enough documents retrieved!", stacklevel=2)
        else:
            hits = hits[:num]

        if self.contain_doc:
            all_contents = [json.loads(self.searcher.doc(hit.docid).raw())["contents"] for hit in hits]
            results = [
                {
                    "title": content.split("\n")[0].strip('"'),
                    "text": "\n".join(content.split("\n")[1:]),
                    "contents": content,
                }
                for content in all_contents
            ]
        else:
            results = load_docs(self.corpus, [hit.docid for hit in hits])

        if return_score:
            return results, scores
        else:
            return results

    def _batch_search(self, query_list: Sequence[str], num: int = None, return_score: bool = True):
        results = []
        scores = []
        for query in query_list:
            item_result, item_score = self._search(query, num, True)
            results.append(item_result)
            scores.append(item_score)
        if return_score:
            return results, scores
        else:
            return results


class DenseRetriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        self.vector_client: Optional[VectorSearchClient] = getattr(config, "vector_client", None)
        self.vector_collection: Optional[str] = getattr(config, "vector_collection", None)

        if self.vector_client is None:
            self.index = faiss.read_index(self.index_path)
            if config.faiss_gpu:
                co = faiss.GpuMultipleClonerOptions()
                co.useFloat16 = True
                co.shard = True
                self.index = faiss.index_cpu_to_all_gpus(self.index, co=co)
            self.corpus = load_corpus(self.corpus_path)
        else:
            self.index = None
            self.corpus = None

        if config.embedding_backend is None:
            raise RuntimeError("DenseRetriever requires an embedding backend")
        self.backend: EmbeddingBackend = config.embedding_backend
        self.batch_size = config.retrieval_batch_size

    def _search(self, query: str, num: int = None, return_score: bool = True):
        if num is None:
            num = self.topk
        query_emb = self.backend.encode([query], is_query=True)

        if self.vector_client is not None:
            hits = self.vector_client.search(query_emb, collection=self.vector_collection, topk=num)[0]
            documents = [hit.get("document") for hit in hits]
            scores = [float(hit.get("score", 0.0)) for hit in hits]
            if return_score:
                return documents, scores
            return documents

        scores, idxs = self.index.search(query_emb, k=num)
        idxs = idxs[0]
        scores = scores[0]
        results = load_docs(self.corpus, idxs)
        if return_score:
            return results, scores.tolist()
        else:
            return results

    def _batch_search(self, query_list: Sequence[str], num: int = None, return_score: bool = True):
        if isinstance(query_list, str):
            query_list = [query_list]
        if num is None:
            num = self.topk

        results = []
        scores = []
        for start_idx in tqdm(
            range(0, len(query_list), self.batch_size),
            desc="Retrieval process: ",
            disable=len(query_list) < 20,
        ):
            query_batch = list(query_list[start_idx : start_idx + self.batch_size])
            batch_emb = self.backend.encode(query_batch, is_query=True)

            if self.vector_client is not None:
                hits_batch = self.vector_client.search(batch_emb, collection=self.vector_collection, topk=num)
                batch_docs = [[hit.get("document") for hit in hits] for hits in hits_batch]
                batch_scores = [
                    [float(hit.get("score", 0.0)) for hit in hits]
                    for hits in hits_batch
                ]
                results.extend(batch_docs)
                scores.extend(batch_scores)
            else:
                batch_scores, batch_idxs = self.index.search(batch_emb, k=num)
                batch_scores = batch_scores.tolist()
                batch_idxs = batch_idxs.tolist()

                flat_idxs = sum(batch_idxs, [])
                batch_results = load_docs(self.corpus, flat_idxs)
                batch_results = [batch_results[i * num : (i + 1) * num] for i in range(len(batch_idxs))]

                results.extend(batch_results)
                scores.extend(batch_scores)

                del batch_idxs, flat_idxs, batch_results
                torch.cuda.empty_cache()

            del batch_emb, query_batch

        if return_score:
            return results, scores
        else:
            return results


# ---------------------------------------------------------------------------
# FastAPI server plumbing
# ---------------------------------------------------------------------------


class Config:
    """
    Minimal config class (simulating your argparse)
    Replace this with your real arguments or load them dynamically.
    """

    def __init__(
        self,
        retrieval_method: str = "bm25",
        retrieval_topk: int = 10,
        index_path: str = "./index/bm25",
        corpus_path: str = "./data/corpus.jsonl",
        dataset_path: str = "./data",
        data_split: str = "train",
        faiss_gpu: bool = True,
        retrieval_model_path: str = "./model",
        retrieval_pooling_method: str = "mean",
        retrieval_query_max_length: int = 256,
        retrieval_use_fp16: bool = False,
        retrieval_batch_size: int = 128,
        embedding_backend: Optional[EmbeddingBackend] = None,
        vector_client: Optional[VectorSearchClient] = None,
        vector_collection: Optional[str] = None,
    ):
        self.retrieval_method = retrieval_method
        self.retrieval_topk = retrieval_topk
        self.index_path = index_path
        self.corpus_path = corpus_path
        self.dataset_path = dataset_path
        self.data_split = data_split
        self.faiss_gpu = faiss_gpu
        self.retrieval_model_path = retrieval_model_path
        self.retrieval_pooling_method = retrieval_pooling_method
        self.retrieval_query_max_length = retrieval_query_max_length
        self.retrieval_use_fp16 = retrieval_use_fp16
        self.retrieval_batch_size = retrieval_batch_size
        self.embedding_backend = embedding_backend
        self.vector_client = vector_client
        self.vector_collection = vector_collection


def get_retriever(config):
    if config.retrieval_method == "bm25":
        return BM25Retriever(config)
    else:
        return DenseRetriever(config)


#####################################
# FastAPI server below
#####################################


class QueryRequest(BaseModel):
    queries: list[str]
    topk: Optional[int] = None
    return_scores: bool = True


app = FastAPI()
config: Optional[Config] = None
retriever: Optional[BaseRetriever] = None


@app.post("/retrieve")
def retrieve_endpoint(request: QueryRequest):
    """
    Endpoint that accepts queries and performs retrieval.

    Input format:
    {
      "queries": ["What is Python?", "Tell me about neural networks."],
      "topk": 3,
      "return_scores": true
    }

    Output format (when return_scores=True，similarity scores are returned):
    {
        "result": [
            [   # Results for each query
                {
                    {"document": doc, "score": score}
                },
                # ... more documents
            ],
            # ... results for other queries
        ]
    }
    """
    if config is None or retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not initialised")

    if not request.topk:
        request.topk = config.retrieval_topk  # fallback to default

    # Perform batch retrieval
    results, scores = retriever.batch_search(
        query_list=request.queries, num=request.topk, return_score=request.return_scores
    )

    # Format response
    resp = []
    for i, single_result in enumerate(results):
        if request.return_scores:
            # If scores are returned, combine them with results
            combined = []
            for doc, score in zip(single_result, scores[i], strict=True):
                combined.append({"document": doc, "score": score})
            resp.append(combined)
        else:
            resp.append(single_result)
    return {"result": resp}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch the retrieval service with shared embeddings.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host binding.")
    parser.add_argument("--port", type=int, default=8000, help="Server port.")
    parser.add_argument("--log_level", type=str, default="info", help="uvicorn log level")
    parser.add_argument(
        "--index_path",
        type=str,
        default="/data2/qirui/verl-tool/data/search_r1/retriever_index/e5_Flat.index",
        help="Path to the FAISS index (for dense retrieval).",
    )
    parser.add_argument(
        "--corpus_path",
        type=str,
        default="/data2/qirui/verl-tool/data/search_r1/retriever_index/wiki-25.jsonl",
        help="Local corpus file used to recover documents by index id.",
    )
    parser.add_argument("--topk", type=int, default=3, help="Number of retrieved passages for one query.")
    parser.add_argument("--retriever_name", type=str, default="e5", help="Retrieval method (e.g., e5, bm25).")
    parser.add_argument(
        "--retriever_model",
        type=str,
        default="intfloat/e5-base-v2",
        help="Model name or path for local dense retriever.",
    )
    parser.add_argument("--faiss_gpu", action="store_true", help="Move FAISS index to GPU when available.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Device for local encoder.")
    parser.add_argument("--use_fp16", action="store_true", help="Use fp16 for local encoder on GPU.")
    parser.add_argument("--max_length", type=int, default=256, help="Maximum token length for encoder inputs.")
    parser.add_argument(
        "--pooling_method",
        type=str,
        default="mean",
        choices=["mean", "cls", "pooler"],
        help="Pooling strategy for sentence embeddings.",
    )
    parser.add_argument("--query_prefix", type=str, default="query: ")
    parser.add_argument("--passage_prefix", type=str, default="passage: ")
    parser.add_argument(
        "--embedding_endpoint",
        type=str,
        default=None,
        help="Optional remote embedding service endpoint (e.g., http://127.0.0.1:9100/embed).",
    )
    parser.add_argument("--embedding_timeout", type=float, default=30.0, help="Timeout for remote embedding calls (s).")
    parser.add_argument(
        "--embedding_max_batch",
        type=int,
        default=128,
        help="Max batch size per request when calling remote embedding service.",
    )
    parser.add_argument(
        "--retrieval_batch_size",
        type=int,
        default=512,
        help="Batch size for processing queries in dense retrieval.",
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
        default="wiki25",
        help="Collection name to use when querying the vector search service.",
    )
    parser.add_argument(
        "--vector_timeout",
        type=float,
        default=30.0,
        help="Timeout (seconds) for vector search requests.",
    )

    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    vector_client = None

    if args.vector_endpoint:
        vector_client = VectorSearchClient(endpoint=args.vector_endpoint, timeout=args.vector_timeout)

    if args.retriever_name.lower() == "bm25":
        embedding_backend = None
    elif args.embedding_endpoint:
        embedding_backend = RemoteEmbeddingBackend(
            endpoint=args.embedding_endpoint,
            timeout=args.embedding_timeout,
            max_batch_size=args.embedding_max_batch,
        )
    else:
        device = _select_device(args.device)
        embedding_backend = LocalEmbeddingBackend(
            model_name=args.retriever_model,
            device=device,
            pooling_method=args.pooling_method,
            max_length=args.max_length,
            use_fp16=bool(args.use_fp16 and device.type == "cuda"),
            query_prefix=args.query_prefix,
            passage_prefix=args.passage_prefix,
        )

    cfg = Config(
        retrieval_method=args.retriever_name,
        index_path=args.index_path,
        corpus_path=args.corpus_path,
        retrieval_topk=args.topk,
        faiss_gpu=args.faiss_gpu,
        retrieval_model_path=args.retriever_model,
        retrieval_pooling_method=args.pooling_method,
        retrieval_query_max_length=args.max_length,
        retrieval_use_fp16=args.use_fp16,
        retrieval_batch_size=args.retrieval_batch_size,
        embedding_backend=embedding_backend,
        vector_client=vector_client,
        vector_collection=args.vector_collection,
    )

    config = cfg
    retriever = get_retriever(cfg)

    if isinstance(retriever, DenseRetriever) and embedding_backend is None:
        raise RuntimeError(
            "Dense retrieval selected but no embedding backend configured. Provide --embedding_endpoint or allow local encoder."
        )

    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)

