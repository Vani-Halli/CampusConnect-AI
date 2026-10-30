import hashlib
import logging
from typing import Any

import numpy as np

from app.services.text_preprocessing import normalize_text

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover
    SentenceTransformer = None

logger = logging.getLogger(__name__)


class SemanticRetriever:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.embeddings: np.ndarray | None = None
        self._corpus_signature: str | None = None
        self._load_attempted = False
        self.load_error: str | None = None

    def _load_model(self) -> None:
        if self._load_attempted:
            return
        self._load_attempted = True
        if SentenceTransformer is None:
            self.load_error = "sentence-transformers is not installed"
            return
        try:
            self.model = SentenceTransformer(self.model_name)
        except Exception as exc:
            self.load_error = str(exc)
            logger.warning("Semantic model %s could not be loaded: %s", self.model_name, exc)
            self.model = None

    def embed(self, texts: list[str]) -> np.ndarray:
        if self.model is None:
            return np.empty((0, 0), dtype=float)
        vectors = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return np.asarray(vectors, dtype=float)

    def search(self, query: str, records: list[dict[str, Any]], top_k: int = 3) -> list[dict[str, Any]]:
        normalized_query = normalize_text(query)
        if not normalized_query or top_k <= 0:
            return []
        self._load_model()
        if self.model is None or not records:
            return []
        texts = [
            normalize_text(
                record.get("searchable_text")
                or " ".join(str(record.get(key, "")) for key in ("primary_question", "answer"))
            )
            for record in records
        ]
        if not texts:
            return []
        signature = hashlib.sha256("\n".join(texts).encode("utf-8")).hexdigest()
        try:
            if self.embeddings is None or signature != self._corpus_signature:
                self.embeddings = np.asarray(
                    self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True),
                    dtype=float,
                )
                self._corpus_signature = signature
            question_embedding = np.asarray(
                self.model.encode(
                    [normalized_query],
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                ),
                dtype=float,
            )
            similarities = np.dot(self.embeddings, question_embedding[0])
        except Exception as exc:
            self.load_error = f"Semantic search failed: {exc}"
            logger.exception("Semantic search failed; TF-IDF fallback will be used")
            self.model = None
            self.embeddings = None
            self._corpus_signature = None
            return []

        if similarities.ndim != 1 or len(similarities) != len(records):
            self.load_error = "Semantic model returned embeddings with an incompatible shape"
            logger.error("%s; TF-IDF fallback will be used", self.load_error)
            self.model = None
            self.embeddings = None
            self._corpus_signature = None
            return []

        candidates = sorted(
            range(len(similarities)),
            key=lambda index: (-float(similarities[index]), index),
        )[:top_k]
        results = []
        for idx in candidates:
            score = float(similarities[int(idx)])
            if not np.isfinite(score) or score <= 0:
                continue
            record = records[int(idx)]
            results.append({
                "record": record,
                "score": round(score, 4),
                "method": "semantic",
            })
        return results
