import json
import hmac
import logging
import os
import re
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal
from pypdf import PdfReader

from app.core.config import settings
from app.database import get_analytics_summary, list_document_chunks, list_documents, record_analytics, record_feedback, remove_document, save_document, save_document_chunks
from app.services.answer_builder import build_answer
from app.services.answerability import is_answerable_hit
from app.services.data_loader import load_bec_corpus
from app.services.faq_service import FAQService
from app.services.llm_service import generate_answer_with_llm
from app.services.query_intent import (
    category_for_query,
    classify_count_query,
    classify_collection_query,
    filter_collection_records,
    unsupported_collection_fact,
)
from app.services.retrieval import SemanticRetriever
from app.services.retrieval_engine import AcademicRetrievalEngine

logger = logging.getLogger(__name__)
app = FastAPI(title="CampusConnect AI", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in settings.allowed_origins.split(",") if origin.strip()],
    allow_origin_regex=(
        r"^http://(?:localhost|127(?:\.\d{1,3}){3}|10(?:\.\d{1,3}){3}|"
        r"192\.168(?:\.\d{1,3}){2}|172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2}):5173$"
        if settings.app_env == "development"
        else None
    ),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

semantic_retriever = SemanticRetriever(model_name=settings.semantic_model)
dataset_path = settings.bec_dataset_path
if not dataset_path.is_absolute():
    dataset_path = (Path(__file__).resolve().parents[1] / dataset_path).resolve()
EVALUATION_REPORT_PATH = Path(__file__).resolve().parents[1] / "reports" / "retrieval_evaluation.json"
corpus_result = load_bec_corpus(dataset_path)
faq_service = FAQService(records=corpus_result.documents)
retrieval_engine = AcademicRetrievalEngine(
    corpus_result.documents,
    semantic_retriever,
    tfidf_min_score=settings.tfidf_min_score,
    semantic_min_score=settings.semantic_min_score,
    rrf_constant=settings.hybrid_rrf_constant,
)
UPLOAD_DIR = Path(settings.upload_dir)
if not UPLOAD_DIR.is_absolute():
    UPLOAD_DIR = (Path(__file__).resolve().parents[1] / UPLOAD_DIR).resolve()
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def _uploaded_chunk_records() -> list[dict[str, Any]]:
    records = []
    for row in list_document_chunks():
        document_id = str(row["chunk_id"])
        title = f"{row['document_name']} - page {row['page_number']}"
        original = {"document_name": row["document_name"], "page_number": row["page_number"], "text": row["content"]}
        records.append({
            "id": document_id,
            "document_id": document_id,
            "category": "Uploaded PDFs",
            "source_file": row["document_name"],
            "title": title,
            "primary_question": None,
            "answer": row["content"],
            "page_reference": row["page_number"],
            "searchable_text": row["content"],
            "original_record": original,
            "source_reference": {
                "document_id": document_id,
                "category": "Uploaded PDFs",
                "source_file": row["document_name"],
                "page": row["page_number"],
                "title": title,
            },
        })
    return records


def _refresh_retrieval_records() -> None:
    retrieval_engine.update_records(corpus_result.documents + _uploaded_chunk_records())


def _split_pdf_page(text: str, chunk_size: int = 1800, overlap: int = 250) -> list[str]:
    normalized = re.sub(r"\s+", " ", text).strip()
    chunks = []
    start = 0
    while start < len(normalized):
        end = min(start + chunk_size, len(normalized))
        if end < len(normalized):
            boundary = normalized.rfind(" ", start + chunk_size // 2, end)
            if boundary > start:
                end = boundary
        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(normalized):
            break
        start = max(start + 1, end - overlap)
    return chunks


def _contextualize_follow_up(question: str, history: list[str]) -> str:
    if not history or category_for_query(question):
        return question

    normalized = question.casefold().strip()
    references_context = (
        re.search(r"\b(?:it|its|they|them|those|these|that|there|one)\b", normalized)
        or re.match(r"^(?:and|also|what about|how about)\b", normalized)
    )
    previous_question = history[-1].strip()[:500]
    if not references_context or not previous_question:
        return question
    return f"{question} {previous_question}"


def _placement_bullet(record: dict[str, Any], prefixes: tuple[str, ...]) -> str:
    answer = build_answer(record)
    prefix_pattern = r"^(?:" + "|".join(re.escape(prefix) for prefix in prefixes) + r")\s*"
    fact = re.sub(prefix_pattern, "", answer, count=1, flags=re.IGNORECASE)
    fact = re.sub(r"^an average package of", "Average package:", fact, count=1, flags=re.IGNORECASE)
    if fact:
        fact = fact[0].upper() + fact[1:]
    return "- " + fact


def _summarize_collection(
    category: str,
    records: list[dict[str, Any]],
    question: str = "",
) -> str:
    if category == "Courses":
        corpus_text = " ".join(
            f"{record.get('question', '')} {record.get('answer', '')}"
            for record in records
        )
        programmes = [
            label
            for pattern, label in (
                (r"\bB\.?E\.?\b", "B.E."),
                (r"\bM\.?Tech\b", "M.Tech"),
                (r"\bPh\.?D\.?\b", "Ph.D."),
                (r"\bMBA\b", "MBA"),
                (r"\bMCA\b", "MCA"),
            )
            if re.search(pattern, corpus_text, flags=re.IGNORECASE)
        ]
        topics = [
            label
            for field, label in (
                ("discipline", "disciplines"),
                ("specialization", "specializations"),
                ("research centre", "research centres"),
                ("duration", "programme duration"),
                ("intake", "intake"),
            )
            if field in corpus_text.casefold()
        ]
        summary = f"The current BEC course FAQs cover {', '.join(programmes)}."
        if topics:
            summary += f" They include information on {', '.join(topics)}."
        return f"{summary} Choose a related FAQ below for a direct answer."

    if category == "Placements":
        def academic_year(record: dict[str, Any]) -> str:
            original = record.get("original_record")
            value = original.get("academic_year") if isinstance(original, dict) else None
            return str(value or "").casefold()

        def record_question(record: dict[str, Any]) -> str:
            return str(record.get("primary_question") or record.get("question") or "").casefold()

        batch_2025 = [
            record
            for record in records
            if academic_year(record) == "2025 batch"
            and re.search(r"\b(?:eligible for placement|students were placed|placement offers)\b", record_question(record))
        ]
        offers_2026 = [
            record
            for record in records
            if academic_year(record) == "2026"
            and "placement offers" in record_question(record)
        ]
        other_figures = [
            record
            for record in records
            if not academic_year(record)
            and re.search(r"\b(?:average package|recruiters)\b", record_question(record))
        ]
        summary_lines = ["BEC's placement information in the current dataset:"]
        if batch_2025:
            summary_lines.extend(
                [
                    "",
                    "2025 batch",
                    *[
                        _placement_bullet(
                            record,
                            (
                                "The placement overview supplied for the 2025 batch lists",
                                "The supplied placement overview lists",
                                "The supplied overview lists",
                            ),
                        )
                        for record in batch_2025
                    ],
                ]
            )
        if offers_2026:
            summary_lines.extend(
                [
                    "",
                    "2026",
                    *[
                        _placement_bullet(
                            record,
                            ("The supplied placement overview shows",),
                        )
                        for record in offers_2026
                    ],
                ]
            )
        if other_figures:
            summary_lines.extend(
                [
                    "",
                    "Other figures (reporting period not specified)",
                    *[
                        _placement_bullet(
                            record,
                            ("The supplied placement overview lists",),
                        )
                        for record in other_figures
                    ],
                ]
            )

        policy_requested = bool(
            re.search(r"\b(?:policy|recruitment|priority slots?|ctc bands?)\b", question, re.IGNORECASE)
        )
        if policy_requested:
            policy_records = [
                record for record in records
                if "organize campus placement recruitment" in record_question(record)
            ]
            if policy_records:
                summary_lines.extend(
                    [
                        "",
                        "Placement policy",
                        *[
                            _placement_bullet(
                                record,
                                ("The supplied placement policy says",),
                            )
                            for record in policy_records
                        ],
                    ]
                )

        if len(summary_lines) > 1:
            summary_lines.extend(
                [
                    "",
                    "Choose a related FAQ below for more detail.",
                ]
            )
            return "\n".join(summary_lines)

    questions = [
        str(record.get("primary_question") or record.get("question") or record.get("title") or "").strip()
        for record in records
    ]
    questions = [question for question in questions if question][:3]
    if questions:
        return (
            f"The current {category.lower()} FAQs cover topics such as {'; '.join(questions)}. "
            "Choose a related FAQ below for a direct answer."
        )
    return (
        f"The current BEC dataset contains {len(records)} {category.lower()} FAQ records. "
        "Choose a related FAQ below to explore the available information."
    )


def _is_placement_overview_record(record: dict[str, Any]) -> bool:
    question = str(record.get("primary_question") or record.get("question") or "").casefold()
    return bool(
        re.search(
            r"\b(?:how many students were eligible for placement|how many students were placed|"
            r"how many placement offers|what average package|how many recruiters|"
            r"how does bec organize campus placement recruitment)\b",
            question,
        )
    )


def require_document_admin(x_admin_key: str | None = Header(default=None)) -> None:
    configured_key = settings.admin_api_key
    if not configured_key:
        raise HTTPException(
            status_code=503,
            detail="Document management is disabled until ADMIN_API_KEY is configured.",
        )
    if not x_admin_key or not hmac.compare_digest(x_admin_key, configured_key):
        raise HTTPException(status_code=403, detail="A valid document administrator key is required.")


_refresh_retrieval_records()


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=2, max_length=500)
    mode: Literal["tfidf", "semantic", "hybrid"] = Field(default="semantic")
    history: list[str] = Field(default_factory=list, max_length=8)
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=10, ge=1, le=50)


class FeedbackRequest(BaseModel):
    question: str = Field(..., min_length=2)
    answer: str = Field(..., min_length=2)
    helpful: bool
    category: str | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict[str, Any]]
    records: list[dict[str, Any]] = Field(default_factory=list)
    total: int = 0
    page: int = 1
    page_size: int = 10
    has_more: bool = False
    intent: str = "specific"
    retrieval_mode: str
    status: str
    query: str
    retrieved_document_ids: list[str]
    category: str | None = None
    retrieval_method: str
    score: float | None = None
    follow_up_questions: list[str]
    mode_label: str
    generation_status: str


class DocumentResponse(BaseModel):
    id: int
    document_name: str
    status: str
    page_count: int | None = None
    uploaded_at: str | None = None
    file_size: int | None = None
    indexed: int = 1


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "app": settings.app_name,
        "mode": settings.retrieval_mode,
        "corpus_documents": len(corpus_result.documents),
        "corpus_files": corpus_result.files_loaded,
        "dataset_name": (corpus_result.dataset_metadata or {}).get("dataset_name", "BEC CampusConnect Dataset"),
        "institution": "Basaveshwar Engineering College, Bagalkote",
        "dataset_source": "User-provided BEC FAQ dataset; source records have not been independently verified",
        "dataset_records_read": corpus_result.records_read,
        "dataset_invalid_records": corpus_result.invalid_records,
        "dataset_duplicate_questions": corpus_result.duplicate_questions,
        "dataset_generated_ids": sum(record["document_id"].startswith("BEC-ROW-") for record in corpus_result.documents),
        "dataset_warnings": corpus_result.warnings,
        "dataset_metadata": corpus_result.dataset_metadata,
        "semantic_model_loaded": semantic_retriever.model is not None,
        "semantic_model_error": semantic_retriever.load_error,
    }


@app.get("/api/evaluation")
def get_retrieval_evaluation() -> dict[str, Any]:
    if not EVALUATION_REPORT_PATH.is_file():
        return {
            "status": "not_run",
            "message": "No saved BEC retrieval evaluation report is available. Run the evaluation script to create it.",
        }

    try:
        report = json.loads(EVALUATION_REPORT_PATH.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        logger.exception("Could not read the saved BEC retrieval evaluation report")
        raise HTTPException(status_code=500, detail="The saved retrieval evaluation report could not be loaded.")

    if (
        not isinstance(report, dict)
        or report.get("dataset") != "bec_campusconnect_dataset.json"
        or report.get("knowledge_documents") != len(corpus_result.documents)
        or not isinstance(report.get("metrics"), dict)
    ):
        return {
            "status": "not_run",
            "message": "The saved evaluation report does not match the active BEC dataset. Run the evaluation script again.",
        }

    metrics: dict[str, dict[str, Any]] = {}
    metric_fields = (
        "top1_accuracy",
        "top3_accuracy",
        "mrr",
        "answerable_acceptance_rate",
        "rejection_accuracy",
        "answerable_queries",
        "rejection_queries",
        "false_acceptances",
    )
    for method, result in report["metrics"].items():
        if not isinstance(result, dict):
            continue
        summary: dict[str, Any] = {"status": result.get("status", "unavailable")}
        details = result.get("metrics")
        if isinstance(details, dict):
            summary.update({field: details[field] for field in metric_fields if field in details})
        metrics[str(method)] = summary

    return {
        "status": "measured",
        "dataset": report["dataset"],
        "knowledge_documents": report["knowledge_documents"],
        "answerable_query_count": report.get("answerable_query_count", 0),
        "rejection_query_count": report.get("rejection_query_count", 0),
        "semantic_model": report.get("semantic_model"),
        "semantic_model_available": report.get("semantic_model_available", False),
        "metrics": metrics,
        "limitations": report.get("limitations", []),
    }


@app.get("/api/faqs")
def get_faqs() -> list[dict[str, Any]]:
    return faq_service.get_all_faqs()


@app.get("/api/categories")
def get_categories() -> list[str]:
    return sorted({str(record.get("category")) for record in retrieval_engine.records if record.get("category")})


@app.get("/api/category-counts")
def category_counts() -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in corpus_result.documents:
        category = str(record.get("category", ""))
        if category:
            counts[category] = counts.get(category, 0) + 1
    return counts


@app.get("/api/knowledge")
def search_knowledge(
    query: str = Query(default="", max_length=500),
    category: str | None = None,
    mode: Literal["tfidf", "semantic", "hybrid"] = "tfidf",
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
) -> dict[str, Any]:
    records = retrieval_engine.records
    hits = []
    if category:
        records = [record for record in records if str(record.get("category", "")).casefold() == category.casefold()]
    if query.strip():
        hits = retrieval_engine.search(query.strip(), mode=mode, top_k=len(records), records=records)
        matching_ids = {hit["document_id"] for hit in hits}
        records = [record for record in records if str(record["document_id"]) in matching_ids]
    rank_by_id = {
        hit["document_id"]: index
        for index, hit in enumerate(hits)
    } if query.strip() else {}
    if query.strip():
        records.sort(key=lambda record: rank_by_id.get(str(record["document_id"]), len(rank_by_id)))
    total = len(records)
    offset = (page - 1) * page_size
    items = [
        {
            "document_id": record["document_id"],
            "category": record["category"],
            "title": record["title"],
            "source_file": record["source_file"],
            "source_reference": record["source_reference"],
            "original_record": record["original_record"],
        }
        for record in records[offset:offset + page_size]
    ]
    return {
        "items": items,
        "total": total,
        "page": page,
        "page_size": page_size,
        "has_more": offset + len(items) < total,
    }


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    if len(question) < 2:
        raise HTTPException(status_code=400, detail="Question is too short.")

    safe_question = re.sub(r"\s+", " ", question)
    count_category = classify_count_query(safe_question)
    if count_category:
        category_records = [
            record for record in corpus_result.documents
            if str(record.get("category", "")).casefold() == count_category.casefold()
        ]
        count = len(category_records)
        count_description = "question-and-answer records"
        answer = (
            f"There {'is' if count == 1 else 'are'} {count} {count_description} "
            f"in the {count_category.lower()} category. This is a count of FAQ records, "
            "not a count of courses, people, or services."
        )
        record_analytics(safe_question, count_category, "count", answer, 1.0, count, 0)
        return ChatResponse(
            answer=answer,
            sources=[],
            records=[],
            total=count,
            page=1,
            page_size=request.page_size,
            has_more=False,
            intent="count",
            retrieval_mode=request.mode,
            status="answered",
            query=safe_question,
            retrieved_document_ids=[],
            category=count_category,
            retrieval_method="complete_collection_count",
            score=None,
            follow_up_questions=[],
            mode_label="Complete collection count",
            generation_status="not_attempted",
        )

    if (
        category_for_query(safe_question) == "Fees"
        and re.search(r"\b(?:fees?|tuition|payment)\b", safe_question, re.IGNORECASE)
        and not re.search(r"\b(?:b\.?\s*e\.?|mba|mca)\b", safe_question, re.IGNORECASE)
    ):
        clarification = (
            "Which program's fee structure do you mean: B.E., MBA, or MCA? "
            "I can look up the fee information available for that program in the BEC dataset."
        )
        record_analytics(safe_question, "Fees", "clarification", clarification, 0.0, 0, 0)
        return ChatResponse(
            answer=clarification,
            sources=[],
            retrieval_mode=request.mode,
            status="unsupported",
            query=safe_question,
            retrieved_document_ids=[],
            category="Fees",
            retrieval_method="clarification",
            score=None,
            follow_up_questions=[],
            mode_label="Clarification needed",
            generation_status="not_attempted",
            intent="specific",
        )

    collection_category = classify_collection_query(safe_question)
    if collection_category:
        matching_records = [
            record for record in retrieval_engine.records
            if str(record.get("category", "")).casefold() == collection_category.casefold()
        ]
        if collection_category == "Placements" and re.search(
            r"\b(?:opportunit(?:y|ies)|overview|results?)\b",
            safe_question,
            flags=re.IGNORECASE,
        ):
            matching_records = [
                record for record in matching_records
                if _is_placement_overview_record(record)
            ]
        matching_records, filter_error = filter_collection_records(
            safe_question,
            collection_category,
            matching_records,
        )
        filter_error = filter_error or unsupported_collection_fact(
            safe_question,
            collection_category,
            matching_records,
        )
        if filter_error:
            return ChatResponse(
                answer=filter_error,
                sources=[],
                retrieval_mode=request.mode,
                status="unsupported",
                query=safe_question,
                retrieved_document_ids=[],
                category=collection_category,
                retrieval_method="collection_filter",
                score=None,
                follow_up_questions=[],
                mode_label="Collection search",
                generation_status="not_attempted",
                total=0,
                page=request.page,
                page_size=request.page_size,
                intent="collection",
            )

        total = len(matching_records)
        start = (request.page - 1) * request.page_size
        page_records = matching_records[start:start + request.page_size]
        if not total:
            return ChatResponse(
                answer=(
                    f"Information about {collection_category.lower()} is not available in the current BEC "
                    "knowledge base. Please contact the appropriate college office to confirm."
                    if not matching_records
                    else f"No {collection_category.lower()} records matched the requested filter."
                ),
                sources=[],
                retrieval_mode=request.mode,
                status="unsupported",
                query=safe_question,
                retrieved_document_ids=[],
                category=collection_category,
                retrieval_method="collection_filter",
                score=None,
                follow_up_questions=[],
                mode_label="Collection search",
                generation_status="not_attempted",
                total=0,
                page=request.page,
                page_size=request.page_size,
                intent="collection",
            )
        source_records = page_records
        sources = [
            {
                "title": record.get("title"),
                "matched_question": record.get("primary_question") or record.get("question") or record.get("title"),
                "category": record.get("category"),
                "document_id": record["document_id"],
                "source_file": record["source_file"],
                "url": record.get("source_url"),
                "verification_status": record.get("verification_status"),
                "verification_note": (
                    record.get("original_record", {}).get("verification_note")
                    if isinstance(record.get("original_record"), dict)
                    else record.get("verification_note")
                ),
                "source_page_label": record.get("source_page_label"),
                "source_reference": record["source_reference"],
            }
            for record in source_records
        ]
        list_answer = _summarize_collection(collection_category, matching_records, safe_question)
        record_analytics(safe_question, collection_category, "collection", list_answer, 1.0, total, 0)
        return ChatResponse(
            answer=list_answer,
            sources=sources,
            records=[
                {
                    "document_id": record["document_id"],
                    "category": record["category"],
                    "title": record["title"],
                    "source_file": record["source_file"],
                    "answer": build_answer(record),
                    "original_record": record["original_record"],
                }
                for record in page_records
            ],
            total=total,
            page=request.page,
            page_size=request.page_size,
            has_more=start + len(page_records) < total,
            intent="collection",
            retrieval_mode=request.mode,
            status="answered",
            query=safe_question,
            retrieved_document_ids=[record["document_id"] for record in page_records],
            category=collection_category,
            retrieval_method="collection",
            score=None,
            follow_up_questions=[
                str(record.get("primary_question") or record.get("question") or record.get("title"))
                for record in matching_records
                if str(record.get("primary_question") or record.get("question") or record.get("title") or "").strip()
            ][:5],
            mode_label="FAQ collection summary",
            generation_status="not_attempted",
        )

    retrieval_question = _contextualize_follow_up(safe_question, request.history)
    query_category = category_for_query(retrieval_question)
    admission_process_query = (
        query_category == "Admissions"
        and bool(re.search(r"\b(?:process|procedure|steps|apply)\b", retrieval_question, flags=re.IGNORECASE))
    )
    generic_contact_query = (
        query_category == "College Contacts"
        and not re.search(
            r"\b(?:placement|exam(?:ination)?|dean|principal|hostel|department|office)\b",
            retrieval_question,
            flags=re.IGNORECASE,
        )
    )
    scoped_records = retrieval_engine.records
    if query_category:
        scoped_records = [
            record for record in retrieval_engine.records
            if str(record.get("category", "")).casefold() == query_category.casefold()
        ]
        if generic_contact_query:
            general_contact_records = [
                record for record in scoped_records
                if re.search(
                    r"\bgeneral (?:contact|enquir|information|details)",
                    str(record.get("primary_question") or record.get("question") or ""),
                    flags=re.IGNORECASE,
                )
            ]
            if general_contact_records:
                scoped_records = general_contact_records
                retrieval_question = f"{retrieval_question} general contact information"
        if not scoped_records:
            fallback_answer = (
                f"Information about {query_category.lower()} is not available in the current BEC "
                "knowledge base. Please contact the appropriate college office to confirm."
            )
            record_analytics(safe_question, query_category, request.mode, fallback_answer, 0.0, 0, 1)
            return ChatResponse(
                answer=fallback_answer,
                sources=[],
                retrieval_mode=request.mode,
                status="unsupported",
                query=safe_question,
                retrieved_document_ids=[],
                category=query_category,
                retrieval_method=request.mode,
                score=0.0,
                follow_up_questions=[],
                mode_label="Information unavailable",
                generation_status="not_attempted",
                intent="specific",
            )
        if admission_process_query:
            process_records = [
                record for record in scoped_records
                if re.search(
                    r"\b(?:entrance exams?|admission routes?|application)\b",
                    " ".join(
                        str(record.get(field) or "")
                        for field in ("question", "primary_question", "title", "answer")
                    ),
                    flags=re.IGNORECASE,
                )
            ]
            if process_records:
                scoped_records = process_records
        unsupported_reason = unsupported_collection_fact(retrieval_question, query_category, scoped_records)
        if unsupported_reason:
            record_analytics(safe_question, query_category, request.mode, unsupported_reason, 0.0, 0, 1)
            return ChatResponse(
                answer=unsupported_reason,
                sources=[],
                retrieval_mode=request.mode,
                status="unsupported",
                query=safe_question,
                retrieved_document_ids=[],
                category=query_category,
                retrieval_method="evidence_validation",
                score=None,
                follow_up_questions=[],
                mode_label="Insufficient evidence",
                generation_status="not_attempted",
                intent="specific",
            )
    chosen_hits = retrieval_engine.search(
        retrieval_question,
        mode=request.mode,
        top_k=12,
        records=scoped_records,
    )
    chosen_hits = [
        hit
        for hit in chosen_hits
        if is_answerable_hit(
            retrieval_question,
            hit,
            request.mode,
            tfidf_min_score=settings.answer_tfidf_min_score,
            semantic_min_score=settings.answer_semantic_min_score,
        )
    ]
    if not chosen_hits:
        fallback_answer = (
            "I couldn't find a reliable answer to that question in my current college FAQ dataset. "
            "This information is not available in the current BEC knowledge base. "
            "Could you rephrase it or ask about admissions, courses, fees, placements, or internships? "
            "For confirmed information, please contact the appropriate Basaveshwar Engineering College office."
        )
        record_analytics(safe_question, None, request.mode, fallback_answer, 0.0, 0, 1)
        return ChatResponse(
            answer=fallback_answer,
            sources=[],
            retrieval_mode=request.mode,
            status="unsupported",
            query=safe_question,
            retrieved_document_ids=[],
            retrieval_method=request.mode,
            score=0.0,
            follow_up_questions=["How can I apply for admission?", "Where can I find the fee structure?", "How can I contact the appropriate college office?"],
            mode_label="Fallback answer",
            generation_status="not_attempted",
            intent="specific",
        )

    unique_hits = []
    seen_answers: set[str] = set()
    seen_questions: set[str] = set()
    for hit in chosen_hits:
        rendered = build_answer(hit["record"])
        question_key = re.sub(
            r"\s+",
            " ",
            str(hit["record"].get("primary_question", "")).casefold(),
        ).strip()
        if rendered.casefold() not in seen_answers and question_key not in seen_questions:
            unique_hits.append((hit, rendered))
            seen_answers.add(rendered.casefold())
            seen_questions.add(question_key)
        if len(unique_hits) == 3:
            break
    top_record = unique_hits[0][0]["record"] if unique_hits else chosen_hits[0]["record"]
    answer_text = unique_hits[0][1]
    sources = [
        {
            "title": hit["record"].get("source_title") or hit["record"].get("title"),
            "matched_question": hit["record"].get("primary_question") or hit["record"].get("question") or hit["record"].get("title"),
            "url": hit["record"].get("source_url"),
            "page": hit["record"].get("page_reference"),
            "verification_status": hit["record"].get("verification_status") or "User-provided information; not independently verified",
            "verification_note": (
                hit["record"].get("original_record", {}).get("verification_note")
                if isinstance(hit["record"].get("original_record"), dict)
                else hit["record"].get("verification_note")
            ),
            "source_page_label": hit["record"].get("source_page_label"),
            "category": hit["record"].get("category"),
            "document_id": hit["document_id"],
            "source_file": hit["record"].get("source_file"),
            "source_reference": hit["record"].get("source_reference"),
        }
        for hit, _rendered in unique_hits[:1]
        if hit["document_id"] in retrieval_engine.by_id
    ]

    context = [
        {"title": source["title"], "answer": rendered}
        for source, (_hit, rendered) in zip(sources, unique_hits)
    ]
    if admission_process_query:
        llm_answer, llm_status = None, "not_attempted"
    else:
        llm_answer, llm_status = generate_answer_with_llm(
            safe_question,
            context,
            provider=settings.llm_provider,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            base_url=settings.llm_base_url,
        )
    if llm_answer and llm_status == "llm_generated":
        answer_text = llm_answer
    if admission_process_query:
        answer_text += (
            " The current FAQ records list admission routes but do not provide "
            "a complete step-by-step admission process."
        )

    follow_ups = []
    seen_follow_ups: set[str] = set()
    for hit in chosen_hits:
        if hit["document_id"] == unique_hits[0][0]["document_id"]:
            continue
        question_text = hit["record"].get("primary_question")
        if isinstance(question_text, str) and question_text.strip():
            question_key = question_text.strip().casefold()
            if question_key not in seen_follow_ups:
                follow_ups.append(question_text.strip())
                seen_follow_ups.add(question_key)
        if len(follow_ups) == 5:
            break

    score = chosen_hits[0].get("score", 0.0)
    retrieval_method = chosen_hits[0].get("method", request.mode)
    record_analytics(safe_question, top_record.get("category"), retrieval_method, answer_text, score, len(chosen_hits), 0)
    return ChatResponse(
        answer=answer_text,
        sources=sources,
        retrieval_mode=request.mode,
        status="answered",
        query=safe_question,
        retrieved_document_ids=[hit["document_id"] for hit, _rendered in unique_hits],
        category=top_record.get("category"),
        retrieval_method=retrieval_method,
        score=score,
        follow_up_questions=follow_ups,
        mode_label={"tfidf": "TF-IDF search", "semantic": "Semantic search", "hybrid": "Hybrid search"}.get(request.mode, request.mode),
        generation_status=llm_status,
        intent="specific",
    )


@app.post("/api/feedback")
def save_feedback(feedback: FeedbackRequest) -> dict[str, str]:
    record_feedback(feedback.question, feedback.answer, feedback.helpful, feedback.category)
    return {"status": "saved"}


@app.get("/api/analytics/summary")
def analytics_summary() -> dict[str, Any]:
    return get_analytics_summary()


@app.get("/api/documents", dependencies=[Depends(require_document_admin)])
def documents() -> list[dict[str, Any]]:
    return list_documents()


@app.post("/api/documents/upload", dependencies=[Depends(require_document_admin)])
def upload_document(file: UploadFile = File(...)) -> dict[str, Any]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="A file name is required.")
    ext = os.path.splitext(file.filename)[1].lower()
    if ext != ".pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    max_upload_size = settings.max_upload_size_mb * 1024 * 1024
    if file.size and file.size > max_upload_size:
        raise HTTPException(status_code=413, detail="File is too large for this upload limit.")
    safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", file.filename)
    contents = file.file.read(max_upload_size + 1)
    if len(contents) > max_upload_size:
        raise HTTPException(status_code=413, detail="File is too large for this upload limit.")
    if not contents.startswith(b"%PDF-"):
        raise HTTPException(status_code=400, detail="The uploaded file does not have a valid PDF signature.")
    try:
        reader = PdfReader(BytesIO(contents))
        pages = len(reader.pages)
        chunks = []
        storage_name = f"{uuid.uuid4().hex}_{safe_name}"
        for page_number, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            for chunk_index, text in enumerate(_split_pdf_page(page_text), start=1):
                chunk_id = f"upload-{storage_name}-{page_number}-{chunk_index}"
                chunks.append({"chunk_id": chunk_id, "page_number": page_number, "content": text})
    except Exception as exc:
        raise HTTPException(status_code=400, detail="The PDF could not be read. Please upload a valid PDF file.") from exc

    if chunks:
        destination = UPLOAD_DIR / storage_name
        destination.write_bytes(contents)
    else:
        storage_name = None
    document_id = save_document(
        safe_name,
        "processed" if chunks else "empty",
        pages,
        len(contents),
        indexed=bool(chunks),
        storage_name=storage_name,
    )
    if not chunks:
        return {"id": document_id, "status": "empty_text", "message": "Document uploaded but no text could be extracted. OCR is required for scanned PDFs.", "page_count": pages}
    save_document_chunks(document_id, chunks)
    _refresh_retrieval_records()
    return {
        "id": document_id,
        "status": "processed",
        "message": "PDF uploaded and searchable text extracted successfully.",
        "page_count": pages,
        "chunk_count": len(chunks),
    }


@app.delete("/api/documents/{document_id}", dependencies=[Depends(require_document_admin)])
def delete_document(document_id: int) -> dict[str, str]:
    storage_name = remove_document(document_id)
    if storage_name is None:
        raise HTTPException(status_code=404, detail="Document not found.")
    if storage_name:
        (UPLOAD_DIR / storage_name).unlink(missing_ok=True)
    _refresh_retrieval_records()
    return {"status": "deleted"}


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "CampusConnect AI backend is running."}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.backend_host,
        port=int(os.environ.get("PORT", settings.backend_port)),
        reload=settings.debug,
    )
