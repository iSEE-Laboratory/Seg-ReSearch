"""HTTP client helper for the shared vector search service."""
from __future__ import annotations

import logging
from typing import List, Sequence

import numpy as np
import requests

logger = logging.getLogger(__name__)


class VectorSearchClient:
    """Tiny client for forwarding embedding batches to the vector search service."""

    def __init__(self, endpoint: str, timeout: float = 30.0) -> None:
        if not endpoint.endswith("/search"):
            endpoint = endpoint.rstrip("/") + "/search"
        self.endpoint = endpoint
        self.timeout = timeout

    def search(
        self,
        embeddings: Sequence[Sequence[float]] | np.ndarray,
        *,
        collection: str,
        topk: int,
    ) -> List[List[dict]]:
        if isinstance(embeddings, np.ndarray):
            if embeddings.ndim != 2:
                raise ValueError("Embeddings must be a 2D array")
            payload_embeddings = embeddings.astype("float32", copy=False).tolist()
        else:
            payload_embeddings = [list(vec) for vec in embeddings]

        payload = {
            "collection": collection,
            "embeddings": payload_embeddings,
            "topk": int(topk),
        }
        try:
            response = requests.post(self.endpoint, json=payload, timeout=self.timeout)
            response.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            logger.error("Vector search request failed: %s", exc)
            raise

        data = response.json()
        if "result" not in data:
            raise ValueError("Vector search service response missing 'result'")
        if not isinstance(data["result"], list):
            raise ValueError("Vector search service response 'result' must be a list")
        return data["result"]

