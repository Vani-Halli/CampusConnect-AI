import logging
import math
from typing import Any

from app.services.faq_service import FAQService
from app.services.retrieval import SemanticRetriever
from app.services.text_preprocessing import normalize_text

logger = logging.getLogger(__name__)


class AcademicRetrievalEngine:
    MODES = {"tfidf", "semantic", "hybrid"}

    def __init__(
        self,
        records: list[dict[str, Any]],
        semantic_retriever: SemanticRetriever,
        tfidf_min_score: float = 0.08,
        semantic_min_score: float = 0.20,
        rrf_constant: int = 60,
    ):
        self.records = records
        self.by_id = {
            str(record["document_id"]): record
            for record in records
            if record.get("document_id") is not None
        }
        self.tfidf = FAQService(records=records)
        self.semantic_retriever = semantic_retriever
        self.tfidf_min_score = tfidf_min_score
        self.semantic_min_score = semantic_min_score
        self.rrf_constant = rrf_constant

    def update_records(self, records: list[dict[str, Any]]) -> None:
        self.records = records
        self.by_id = {
            str(record["document_id"]): record
            for record in records
            if record.get("document_id") is not None
        }
        self.tfidf = FAQService(records=records)

    def search(
        self,
        query: str,
        mode: str = "tfidf",
        top_k: int = 3,
        apply_threshold: bool = True,
        records: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        if mode not in self.MODES:
            raise ValueError(f"Unsupported retrieval mode: {mode}")
        normalized_query = normalize_text(query)
        if not normalized_query or top_k <= 0:
            return []
        search_records = self.records if records is None else records
        threshold = self.tfidf_min_score if apply_threshold else 0.0
        keyword_hits = self._with_ids(
            self.tfidf.search_tfidf(
                normalized_query,
                top_k=top_k,
                threshold=threshold,
                records=records,
            ),
            "tfidf",
            search_records,
        )

        if mode == "tfidf":
            return keyword_hits

        try:
            semantic_candidates = self.semantic_retriever.search(
                normalized_query,
                search_records,
                top_k=top_k,
            )
        except Exception:
            logger.exception("Semantic retrieval failed; using TF-IDF candidates")
            semantic_candidates = []
        if apply_threshold:
            semantic_candidates = [
                hit for hit in semantic_candidates
                if self._valid_score(hit) and float(hit["score"]) >= self.semantic_min_score
            ]
        semantic_hits = self._with_ids(semantic_candidates, "semantic", search_records)

        if mode == "semantic":
            return semantic_hits or self._with_ids(keyword_hits, "tfidf_fallback", search_records)
        if not semantic_hits:
            return self._with_ids(keyword_hits, "tfidf_fallback", search_records)
        if not keyword_hits:
            return semantic_hits[:top_k]

        fused: dict[tuple[str, object], dict[str, Any]] = {}
        first_seen_order = 0
        for ranked_hits in (keyword_hits, semantic_hits):
            for rank, hit in enumerate(ranked_hits, start=1):
                key = self._dedupe_key(hit)
                if key not in fused:
                    fused[key] = {
                        **hit,
                        "score": 0.0,
                        "component_scores": {},
                        "_first_seen_order": first_seen_order,
                    }
                    first_seen_order += 1
                fused[key]["component_scores"][hit["method"]] = hit["score"]
                fused[key]["score"] += 1 / (self.rrf_constant + rank)
        ranked = sorted(
            fused.values(),
            key=lambda hit: (-hit["score"], hit["_first_seen_order"]),
        )[:top_k]
        for hit in ranked:
            hit["method"] = "hybrid_rrf"
            hit.pop("_first_seen_order", None)
        return ranked

    @staticmethod
    def _valid_score(hit: Any) -> bool:
        if not isinstance(hit, dict):
            return False
        try:
            return math.isfinite(float(hit.get("score")))
        except (TypeError, ValueError):
            return False

    @classmethod
    def _with_ids(
        cls,
        hits: list[dict[str, Any]],
        method: str,
        allowed_records: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        allowed_record_ids = (
            {id(record) for record in allowed_records}
            if allowed_records is not None
            else None
        )
        prepared = []
        for hit in hits:
            if not isinstance(hit, dict) or not isinstance(hit.get("record"), dict):
                continue
            record = hit["record"]
            if (
                (allowed_record_ids is not None and id(record) not in allowed_record_ids)
                or not cls._valid_score(hit)
            ):
                continue
            document_id = record.get("document_id", record.get("id"))
            prepared.append({
                **hit,
                "document_id": str(document_id) if document_id is not None else "",
                "score": float(hit["score"]),
                "method": method,
            })
        return prepared

    @staticmethod
    def _dedupe_key(hit: dict[str, Any]) -> tuple[str, object]:
        document_id = hit.get("document_id")
        if document_id:
            return "document_id", document_id
        return "record_identity", id(hit["record"])