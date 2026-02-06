"""Standalone embedding service for sharing text encoders across retrievers."""
from __future__ import annotations

import argparse
import logging
from typing import Iterable, List, Literal, Optional

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)


def _select_device(device: str) -> torch.device:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available; falling back to CPU")
        device = "cpu"
    return torch.device(device)


def _chunk(iterable: List[str], size: int) -> Iterable[List[str]]:
    for start in range(0, len(iterable), size):
        yield iterable[start : start + size]


def _pool_embeddings(model_output, attention_mask, pooling: str) -> torch.Tensor:
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


class TextEncoder:
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

        logger.info("Loading embedding model %s on %s", model_name, device)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.model.to(device)
        if self.use_fp16:
            self.model = self.model.half()
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)

    @torch.no_grad()
    def encode(self, texts: List[str], *, mode: Literal["query", "passage"], normalize: bool) -> np.ndarray:
        prepared = self._prepare_inputs(texts, mode)
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
        if normalize:
            embeddings = torch.nn.functional.normalize(embeddings, dim=-1)

        return embeddings.detach().cpu().to(dtype=torch.float32).numpy()

    def _prepare_inputs(self, texts: List[str], mode: Literal["query", "passage"]) -> List[str]:
        if "e5" in self.model_name.lower():
            prefix = self.query_prefix if mode == "query" else self.passage_prefix
            return [f"{prefix}{text}" for text in texts]
        if "bge" in self.model_name.lower() and mode == "query":
            return [
                f"Represent this sentence for searching relevant passages: {text}" for text in texts
            ]
        return texts


class EmbeddingRequest(BaseModel):
    texts: List[str]
    mode: Literal["query", "passage"] = "query"
    normalize: bool = True
    batch_size: Optional[int] = None


class EmbeddingService:
    def __init__(self, encoder: TextEncoder, max_batch: int) -> None:
        self.encoder = encoder
        self.max_batch = max_batch

    def embed(self, request: EmbeddingRequest) -> List[List[float]]:
        if not request.texts:
            raise HTTPException(status_code=400, detail="'texts' must be a non-empty list")
        batch_size = request.batch_size or self.max_batch
        if batch_size <= 0:
            raise HTTPException(status_code=400, detail="batch_size must be positive")

        outputs: List[np.ndarray] = []
        for chunk in _chunk(request.texts, batch_size):
            embeddings = self.encoder.encode(chunk, mode=request.mode, normalize=request.normalize)
            outputs.append(embeddings)
        stacked = np.concatenate(outputs, axis=0)
        if stacked.shape[0] != len(request.texts):
            raise RuntimeError("Embedding size mismatch")
        return stacked.tolist()


app = FastAPI()
service: Optional[EmbeddingService] = None


@app.post("/embed")
def embed_endpoint(request: EmbeddingRequest):
    if service is None:
        raise HTTPException(status_code=503, detail="Embedding service not initialised")
    embeddings = service.embed(request)
    return {"embeddings": embeddings}


@app.get("/health")
def health_check():
    return {"status": "ok"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Shared embedding service for retrievers.")
    parser.add_argument("--model_name", type=str, default="intfloat/e5-base-v2")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9100)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--pooling_method", type=str, default="mean", choices=["mean", "cls", "pooler"])
    parser.add_argument("--use_fp16", action="store_true")
    parser.add_argument("--max_batch_size", type=int, default=128)
    parser.add_argument(
        "--query_prefix",
        type=str,
        default="query: ",
        help="Prefix applied to queries for E5-family models.",
    )
    parser.add_argument(
        "--passage_prefix",
        type=str,
        default="passage: ",
        help="Prefix applied to passages for E5-family models.",
    )
    parser.add_argument("--log_level", type=str, default="info")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    device = _select_device(args.device)
    use_fp16 = bool(args.use_fp16 and device.type == "cuda")

    encoder = TextEncoder(
        model_name=args.model_name,
        device=device,
        max_length=args.max_length,
        pooling_method=args.pooling_method,
        use_fp16=use_fp16,
        query_prefix=args.query_prefix,
        passage_prefix=args.passage_prefix,
    )

    global service
    service = EmbeddingService(encoder, max_batch=args.max_batch_size)

    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()

