"""Real embedding provider adapter using OpenRouter (openai/text-embedding-3-small)."""
from __future__ import annotations

import math
import os
import time
from typing import Any, Protocol

import requests


class Embedder(Protocol):
    model_name: str
    dimensions: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text, in the same order."""
        ...

    def embed_query(self, question: str) -> list[float]:
        """Return a query vector compatible with the stored document vectors."""
        ...


def _validate_vector(values: list[float], expected_dim: int) -> list[float]:
    """Validate finite numeric vector matching expected dimensionality."""
    if len(values) != expected_dim:
        raise RuntimeError(f"Embedding dimension mismatch: expected {expected_dim}, got {len(values)}.")
    norm = math.sqrt(sum(v * v for v in values))
    if norm <= 0.0 or not math.isfinite(norm):
        raise RuntimeError("Received zero or non-finite embedding vector from provider.")
    return [float(v) for v in values]


class OpenRouterEmbedder:
    """OpenRouter embeddings API adapter; raises explicitly on error instead of mocking."""

    ENDPOINT = "https://openrouter.ai/api/v1/embeddings"

    def __init__(
        self,
        model_name: str = "openai/text-embedding-3-small",
        dimensions: int = 1536,
        batch_size: int = 64,
        max_retries: int = 4,
    ) -> None:
        api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is missing in environment/.env. Cannot initialize real embedder."
            )
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")

        self.provider = "openrouter"
        self.model_name = model_name
        self.dimensions = dimensions
        self.batch_size = batch_size
        self.max_retries = max_retries
        self._api_key = api_key
        self._session = requests.Session()

        # Exact telemetry counters from API responses
        self.doc_api_calls = 0
        self.query_api_calls = 0
        self.doc_texts_embedded = 0
        self.query_texts_embedded = 0
        self.doc_tokens = 0
        self.query_tokens = 0
        self.doc_cost_usd = 0.0
        self.query_cost_usd = 0.0
        self.embed_seconds = 0.0
        self.resolved_model = ""
        self.upstream_provider = ""

    def _request_embeddings(self, inputs: list[str]) -> tuple[list[list[float]], int, float]:
        last_error: Exception | None = None
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "model": self.model_name,
            "input": inputs,
        }
        for attempt in range(self.max_retries):
            t0 = time.perf_counter()
            try:
                resp = self._session.post(self.ENDPOINT, headers=headers, json=payload, timeout=45)
                self.embed_seconds += time.perf_counter() - t0
                if resp.status_code >= 400:
                    raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
                data = resp.json()
                items = sorted(data["data"], key=lambda item: item["index"])
                if len(items) != len(inputs):
                    raise RuntimeError(f"Embedding count mismatch: expected {len(inputs)}, got {len(items)}.")
                vectors = [_validate_vector(list(item["embedding"]), self.dimensions) for item in items]
                usage = data.get("usage") or {}
                tokens = int(usage.get("total_tokens") or usage.get("prompt_tokens") or 0)
                cost = float(usage.get("cost") or 0.0)
                if not self.resolved_model:
                    self.resolved_model = str(data.get("model", self.model_name))
                if not self.upstream_provider:
                    self.upstream_provider = str(data.get("provider", ""))
                return vectors, tokens, cost
            except Exception as exc:
                last_error = exc
                if attempt < self.max_retries - 1:
                    time.sleep(min(2.0 ** attempt, 8.0))
        raise RuntimeError(
            f"OpenRouter embedding request failed after {self.max_retries} attempts: {last_error}"
        ) from last_error

    def embed_documents_with_usage(self, texts: list[str]) -> tuple[list[list[float]], dict[str, Any]]:
        """Embed documents and return per-call token and USD cost telemetry."""
        if not texts:
            return [], {"texts": 0, "tokens": 0, "cost_usd": 0.0, "seconds": 0.0}
        all_vectors: list[list[float]] = []
        call_tokens = 0
        call_cost = 0.0
        t0 = time.perf_counter()
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            vecs, toks, cost = self._request_embeddings(batch)
            all_vectors.extend(vecs)
            self.doc_api_calls += 1
            self.doc_texts_embedded += len(batch)
            self.doc_tokens += toks
            self.doc_cost_usd += cost
            call_tokens += toks
            call_cost += cost
        elapsed = time.perf_counter() - t0
        return all_vectors, {
            "texts": len(texts),
            "tokens": call_tokens,
            "cost_usd": round(call_cost, 8),
            "seconds": round(elapsed, 3),
        }

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors, _ = self.embed_documents_with_usage(texts)
        return vectors

    def embed_query(self, question: str) -> list[float]:
        if not question.strip():
            raise ValueError("Query text must not be empty.")
        vecs, toks, cost = self._request_embeddings([question])
        self.query_api_calls += 1
        self.query_texts_embedded += 1
        self.query_tokens += toks
        self.query_cost_usd += cost
        return vecs[0]

    def stats(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "upstream_provider": self.upstream_provider or "OpenAI",
            "model_name": self.model_name,
            "resolved_model": self.resolved_model or "text-embedding-3-small",
            "dimensions": self.dimensions,
            "doc_query_symmetry": "symmetric (identical normalization & prompt formatting; no task prefix)",
            "doc_api_calls": self.doc_api_calls,
            "query_api_calls": self.query_api_calls,
            "doc_texts_embedded": self.doc_texts_embedded,
            "query_texts_embedded": self.query_texts_embedded,
            "doc_tokens": self.doc_tokens,
            "query_tokens": self.query_tokens,
            "doc_cost_usd": round(self.doc_cost_usd, 8),
            "query_cost_usd": round(self.query_cost_usd, 8),
            "total_cost_usd": round(self.doc_cost_usd + self.query_cost_usd, 8),
            "embed_seconds": round(self.embed_seconds, 3),
        }


MyEmbedder = OpenRouterEmbedder
