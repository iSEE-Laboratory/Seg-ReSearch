"""Client utilities for communicating with the remote embedding service."""
from __future__ import annotations

import logging
from typing import Iterable, List, Sequence

import numpy as np
import requests

logger = logging.getLogger(__name__)


class RemoteEmbeddingClient:
    """HTTP client that queries the shared embedding service.

    Parameters
    ----------
    endpoint:
        Base URL of the embedding service (e.g. ``http://localhost:9100/embed``).
    timeout:
        Timeout (seconds) for each HTTP request.
    max_batch_size:
        Maximum number of texts to send per request. Larger inputs are chunked
        into multiple requests transparently.
    """

    def __init__(self, endpoint: str, timeout: float = 30.0, max_batch_size: int = 128) -> None:
        if not endpoint.endswith("/embed"):
            endpoint = endpoint.rstrip("/") + "/embed"
        self.endpoint = endpoint
        self.timeout = timeout
        self.max_batch_size = max_batch_size

    def encode(self, texts: Sequence[str], *, is_query: bool) -> np.ndarray:
        if isinstance(texts, str):
            raise TypeError("'texts' must be a sequence of strings, not a single string")
        texts = list(texts)
        if not texts:
            return np.zeros((0, 0), dtype="float32")

        mode = "query" if is_query else "passage"
        embeddings: List[np.ndarray] = []
        for start in range(0, len(texts), self.max_batch_size):
            chunk = texts[start : start + self.max_batch_size]
            payload = {"texts": chunk, "mode": mode}
            try:
                response = requests.post(self.endpoint, json=payload, timeout=self.timeout)
                response.raise_for_status()
            except Exception as exc:  # noqa: BLE001
                logger.error("Embedding request failed: %s", exc)
                raise

            data = response.json()
            if "embeddings" not in data:
                raise ValueError("Embedding service response is missing 'embeddings'")
            embed_chunk = np.asarray(data["embeddings"], dtype="float32")
            if embed_chunk.ndim != 2:
                raise ValueError("Embeddings must be a 2D array")
            embeddings.append(embed_chunk)

        joined = np.concatenate(embeddings, axis=0)
        if joined.shape[0] != len(texts):
            raise ValueError("Embedding count mismatch between request and response")
        return joined


def encode_iter(client: RemoteEmbeddingClient, texts: Iterable[str], *, is_query: bool, batch_size: int) -> np.ndarray:
    """Convenience helper to encode an arbitrary iterator in batches.

    Returns a stacked numpy array with ``dtype=float32``.
    """

    buffer: List[str] = []
    outputs: List[np.ndarray] = []
    for text in texts:
        buffer.append(text)
        if len(buffer) >= batch_size:
            outputs.append(client.encode(buffer, is_query=is_query))
            buffer.clear()
    if buffer:
        outputs.append(client.encode(buffer, is_query=is_query))
    if not outputs:
        return np.zeros((0, 0), dtype="float32")
    return np.concatenate(outputs, axis=0)

