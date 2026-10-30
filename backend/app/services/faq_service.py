import json
import re
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.services.text_preprocessing import normalize_text, tokenize_text


_QUERY_FRAME_PATTERN = re.compile(
    r"^(?:what\s+(?:is|are|was|were|does|do|can|could|would)\s+"
    r"|how\s+(?:many|much|long|often|can|do|does)\s+"
    r"|(?:is|are|was|were|do|does|can|could|would)\s+(?:the|a|an)\s+)"
)


class FAQService:
    def __init__(self, dataset_path: str | None = None, records: list[dict[str, Any]] | None = None):
        self.dataset_path = Path(dataset_path) if dataset_path else None
        self.records = records if records is not None else self._load_dataset()
        self._vectorizer = None
        self._matrix = None
        self._search_texts: list[str] = []
        self._record_indices: list[int] = []
        self._build_index()

    def _load_dataset(self) -> list[dict[str, Any]]:
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for record in data:
            if not record.get("answer"):
                raise ValueError(f"FAQ answer missing for record {record.get('id')}")
            if not record.get("source_title") or not record.get("source_url"):
                raise ValueError(f"Missing source metadata for record {record.get('id')}")
            if record.get("category") not in {
                "Admissions",
                "Courses",
                "Fees",
                "Scholarships",
                "Internships",
                "Placements",
                "Examinations",
                "Academic Calendar",
                "Student Services",
            }:
                raise ValueError(f"Invalid category for record {record.get('id')}: {record.get('category')}")
        return data

    def _build_index(self) -> None:
        if not self.records:
            self._vectorizer = None
            self._matrix = None
            self._search_texts = []
            self._record_indices = []
            return
        texts: list[str] = []
        self._record_indices = []
        for record_index, record in enumerate(self.records):
            question = record.get("question") or record.get("primary_question") or record.get("title")
            if not isinstance(question, str) or not question.strip():
                question = record.get("searchable_text")
            normalized_question = normalize_text(question)
            if not normalized_question:
                continue
            texts.append(normalized_question)
            self._record_indices.append(record_index)

        self._search_texts = texts
        if not texts:
            self._vectorizer = None
            self._matrix = None
            return

        self._vectorizer = TfidfVectorizer(
            tokenizer=tokenize_text,
            token_pattern=None,
            lowercase=False,
            ngram_range=(1, 2),
        )
        try:
            self._matrix = self._vectorizer.fit_transform(texts)
        except ValueError as exc:
            if "empty vocabulary" not in str(exc).casefold():
                raise
            self._vectorizer = None
            self._matrix = None
            self._search_texts = []
            self._record_indices = []

    def search_tfidf(
        self,
        question: str,
        top_k: int = 3,
        threshold: float = 0.08,
        records: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        normalized_question = normalize_text(question)
        if not normalized_question or top_k <= 0:
            return []
        if self._vectorizer is None or self._matrix is None:
            return []
        query_vector = self._vectorizer.transform([normalized_question])
        if query_vector.nnz == 0:
            return []
        sims = cosine_similarity(query_vector, self._matrix).ravel()
        ranked = sorted(range(len(sims)), key=lambda index: (-float(sims[index]), index))
        analyzer = self._vectorizer.build_analyzer()
        query_for_overlap = _QUERY_FRAME_PATTERN.sub("", normalized_question)
        query_terms = {
            term for term in analyzer(query_for_overlap)
            if " " not in term and term not in {"bec", "basaveshwar", "engineering", "college"}
        }
        minimum_overlap = min(2, len(query_terms)) if query_terms else 1
        allowed_record_ids = {id(record) for record in records} if records is not None else None
        results: list[dict[str, Any]] = []
        for vector_index in ranked:
            score = float(sims[vector_index])
            if score < threshold:
                break
            record = self.records[self._record_indices[vector_index]]
            if allowed_record_ids is not None and id(record) not in allowed_record_ids:
                continue
            record_terms = {
                term for term in analyzer(self._search_texts[vector_index])
                if " " not in term
            }
            if len(query_terms & record_terms) < minimum_overlap:
                continue
            results.append({
                "record": record,
                "score": round(score, 4),
                "method": "tfidf",
            })
            if len(results) >= top_k:
                break
        return results

    def get_categories(self) -> list[str]:
        categories = sorted({record["category"] for record in self.records})
        return categories

    def get_all_faqs(self) -> list[dict[str, Any]]:
        return self.records

    def evaluate_question(self, question: str, mode: str = "tfidf") -> dict[str, Any]:
        if mode == "tfidf":
            hits = self.search_tfidf(question)
        else:
            hits = self.search_tfidf(question)
        if not hits:
            return {
                "answer": "I could not find enough relevant information in the current knowledge base for that question.",
                "sources": [],
                "status": "unsupported",
                "retrieval_mode": mode,
            }
        top_record = hits[0]["record"]
        return {
            "answer": top_record["answer"],
            "sources": [{
                "title": top_record.get("source_title"),
                "url": top_record.get("source_url"),
                "page": top_record.get("page_reference"),
                "verification_status": top_record.get("verification_status"),
            }],
            "status": "answered",
            "retrieval_mode": mode,
            "score": hits[0]["score"],
        }
