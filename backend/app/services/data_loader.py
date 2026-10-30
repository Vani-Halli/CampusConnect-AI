import json
import logging
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

EXPECTED_FILES = (
    "academic_calendar.json",
    "attendance_policy.json",
    "campus_services.json",
    "courses.json",
    "exam_schedule.json",
    "faculty.json",
    "faq.json",
    "holidays.json",
    "hostel.json",
    "placement_policy.json",
)

CATEGORY_SEARCH_TERMS = {
    "faculty": "faculty professor teacher instructor teaches teaching academic staff",
    "courses": "course class subject curriculum credit semester",
    "attendance_policy": "attendance attendance requirement minimum present absence policy",
    "exam_schedule": "exam examination test assessment timetable schedule date time",
    "academic_calendar": "academic calendar term semester event schedule date",
    "placement_policy": "placement eligibility recruitment hiring job career policy",
    "campus_services": "campus service facility office support opening hours timing",
    "holidays": "holiday vacation closure day date",
    "hostel": "hostel dormitory residence accommodation rules",
    "faq": "frequently asked question answer",
}


@dataclass
class CorpusLoadResult:
    documents: list[dict[str, Any]]
    warnings: list[str]
    files_loaded: list[str]
    dataset_metadata: dict[str, Any] | None = None
    records_read: int = 0
    invalid_records: int = 0
    duplicate_questions: int = 0
    records_with_replacement_characters: int = 0


def load_bec_corpus(dataset_path: Path) -> CorpusLoadResult:
    if not dataset_path.is_absolute():
        dataset_path = (Path(__file__).resolve().parents[2] / dataset_path).resolve()
    if not dataset_path.is_file():
        raise FileNotFoundError(f"BEC chatbot dataset not found: {dataset_path}")

    try:
        payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read BEC chatbot dataset {dataset_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"BEC chatbot dataset must contain a JSON object with a 'records' array: {dataset_path}")
    rows = payload.get("records")
    if not isinstance(rows, list):
        raise ValueError(f"BEC chatbot dataset is missing its 'records' array: {dataset_path}")

    warnings: list[str] = []
    documents: list[dict[str, Any]] = []
    question_counts: Counter[str] = Counter()
    invalid_records = 0
    duplicate_questions = 0
    generated_ids = 0
    seen_ids: set[str] = set()
    reserved_ids = {
        row["id"].strip()
        for row in rows
        if isinstance(row, dict) and isinstance(row.get("id"), str) and row["id"].strip()
    }

    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            invalid_records += 1
            warnings.append(f"Skipped BEC record {index}: expected a JSON object")
            continue
        question = row.get("question")
        answer = row.get("answer")
        if not isinstance(question, str) or not question.strip():
            invalid_records += 1
            warnings.append(f"Skipped BEC record {index}: missing or invalid question")
            continue
        if not isinstance(answer, str) or not answer.strip():
            invalid_records += 1
            warnings.append(f"Skipped BEC record {index}: missing or invalid answer")
            continue
        category = row.get("category")
        if not isinstance(category, str) or not category.strip():
            invalid_records += 1
            warnings.append(f"Skipped BEC record {index}: missing or invalid category")
            continue

        searchable_question = re.sub(r"\s+", " ", question).strip()
        searchable_answer = re.sub(r"\s+", " ", answer).strip()
        question_key = searchable_question.casefold()
        if question_counts[question_key]:
            duplicate_questions += 1
        question_counts[question_key] += 1

        source_id = row.get("id")
        if isinstance(source_id, str) and source_id.strip():
            document_id = source_id.strip()
        else:
            document_id = f"BEC-ROW-{index:03d}"
            suffix = index
            while document_id in seen_ids or document_id in reserved_ids:
                suffix += 1
                document_id = f"BEC-ROW-{suffix:03d}"
            generated_ids += 1
        if document_id in seen_ids:
            invalid_records += 1
            warnings.append(f"Skipped BEC record {index}: duplicate record id {document_id!r}")
            continue
        seen_ids.add(document_id)
        institution = row.get("institution")
        if not isinstance(institution, str) or not institution.strip():
            institution = "Basaveshwar Engineering College, Bagalkote"
        source_url = row.get("source_url")
        if not isinstance(source_url, str) or not source_url.strip():
            source_url = None
        source_page_label = row.get("source_page_label")
        if not isinstance(source_page_label, str) or not source_page_label.strip():
            source_page_label = None
        source_status = row.get("source_status")
        if not isinstance(source_status, str) or not source_status.strip():
            source_status = "User-provided information; not independently verified"
        dataset_name = payload.get("dataset_name")
        if not isinstance(dataset_name, str) or not dataset_name.strip():
            dataset_name = "BEC CampusConnect Dataset"
        documents.append({
            "id": document_id,
            "document_id": document_id,
            "category": category,
            "institution": institution,
            "source_file": dataset_path.name,
            "source_title": source_page_label or dataset_name,
            "source_url": source_url,
            "source_page_label": source_page_label,
            "verification_status": source_status,
            "title": question,
            "primary_question": question,
            "question": question,
            "answer": answer,
            "searchable_text": f"{searchable_question} {searchable_answer} {category}",
            "original_record": row,
            "source_reference": {
                "document_id": document_id,
                "category": category,
                "source_file": dataset_path.name,
                "title": question,
                "url": source_url,
                "page_label": source_page_label,
                "institution": institution,
                "verification_status": source_status,
                "academic_year": row.get("academic_year"),
                "verification_note": row.get("verification_note"),
            },
        })

    declared_count = payload.get("record_count")
    if isinstance(declared_count, int) and declared_count != len(rows):
        warnings.append(
            f"Dataset metadata declares {declared_count} records, but the records array contains {len(rows)}"
        )
    if invalid_records:
        warnings.append(f"Skipped {invalid_records} invalid record(s) out of {len(rows)}")
    if duplicate_questions:
        warnings.append(
            f"Found {duplicate_questions} repeated question(s); all records with unique IDs were retained"
        )
    if generated_ids:
        warnings.append(f"Generated local IDs for {generated_ids} records without a source ID")
    if not documents:
        raise ValueError(f"BEC chatbot dataset contains no valid question-and-answer records: {dataset_path}")
    for warning in warnings:
        logger.warning(warning)
    logger.info(
        "Loaded %d BEC FAQ records from %s (%d repeated questions, %d invalid)",
        len(documents),
        dataset_path,
        duplicate_questions,
        invalid_records,
    )
    return CorpusLoadResult(
        documents=documents,
        warnings=warnings,
        files_loaded=[dataset_path.name] if documents else [],
        dataset_metadata={
            key: payload[key]
            for key in ("dataset_name", "version", "created_date", "record_count", "categories", "important_notes", "coverage_gaps")
            if key in payload
        },
        records_read=len(rows),
        invalid_records=invalid_records,
        duplicate_questions=duplicate_questions,
        records_with_replacement_characters=0,
    )


def _records_from_payload(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("records", "documents", "data", "items"):
            nested = payload.get(key)
            if isinstance(nested, (list, dict)):
                return _records_from_payload(nested)
        return [payload]
    return []


def _value_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return "; ".join(
            f"{key.replace('_', ' ')}: {_value_text(item)}"
            for key, item in value.items()
            if _value_text(item)
        )
    if isinstance(value, (list, tuple, set)):
        return ", ".join(text for item in value if (text := _value_text(item)))
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value).strip()


def _searchable_text(record: dict[str, Any]) -> str:
    parts = []
    for key, value in record.items():
        if key in {"document_id", "id"}:
            continue
        text = _value_text(value)
        if text:
            parts.append(f"{key.replace('_', ' ')}: {text}")
    return ". ".join(parts)


def _title(record: dict[str, Any]) -> str:
    for key in ("title", "question", "primary_question", "name", "course", "subject", "service", "event", "holiday", "rule", "policy"):
        value = _value_text(record.get(key))
        if value:
            return value
    return "Academic information"


def load_academic_corpus(
    dataset_dir: Path,
    fallback_faq_file: Path | None = None,
) -> CorpusLoadResult:
    warnings: list[str] = []
    files_loaded: list[str] = []
    documents: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    if not dataset_dir.is_dir():
        warnings.append(f"Academic dataset directory not found: {dataset_dir}")
        paths: list[Path] = []
    else:
        paths = sorted(dataset_dir.glob("*.json"))
        present_names = {path.name for path in paths}
        for expected in EXPECTED_FILES:
            if expected not in present_names:
                warnings.append(f"Dataset file missing: {dataset_dir / expected}")

    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            warnings.append(f"Could not read {path.name}: {exc}")
            continue

        records = _records_from_payload(payload)
        if not records:
            warnings.append(f"No records found in {path.name}")
            continue

        category = "FAQ" if path.stem == "faq" else path.stem.replace("_", " ").title()
        loaded_from_file = False
        for row_number, record in enumerate(records, start=1):
            if not isinstance(record, dict):
                warnings.append(f"Skipped non-object record {row_number} in {path.name}")
                continue
            document_id = record.get("document_id") or record.get("id")
            if document_id is None or not str(document_id).strip():
                warnings.append(f"Skipped record {row_number} in {path.name}: missing document_id")
                continue
            document_id = str(document_id).strip()
            if document_id in seen_ids:
                warnings.append(f"Skipped duplicate document_id {document_id!r} in {path.name}")
                continue
            text = _searchable_text(record)
            if not text:
                warnings.append(f"Skipped empty searchable record {document_id!r} in {path.name}")
                continue
            category_terms = CATEGORY_SEARCH_TERMS.get(path.stem, "")
            if category_terms:
                text = f"{text}. {category_terms}"

            title = _title(record)
            primary_question = record.get("primary_question") or record.get("question")
            document = {
                **record,
                "id": record.get("id", document_id),
                "document_id": document_id,
                "category": record.get("category") or category,
                "source_file": path.name,
                "title": title,
                "primary_question": primary_question,
                "searchable_text": text,
                "original_record": record,
                "source_reference": {
                    "document_id": document_id,
                    "category": record.get("category") or category,
                    "source_file": path.name,
                    "title": title,
                },
            }
            documents.append(document)
            seen_ids.add(document_id)
            loaded_from_file = True
        if loaded_from_file:
            files_loaded.append(path.name)

    if not documents and fallback_faq_file and fallback_faq_file.is_file():
        warnings.append("Academic corpus is unavailable; using the legacy demonstration FAQ dataset.")
        try:
            fallback_records = _records_from_payload(json.loads(fallback_faq_file.read_text(encoding="utf-8")))
            for row_number, record in enumerate(fallback_records, start=1):
                if not isinstance(record, dict):
                    continue
                document_id = record.get("document_id") or record.get("id")
                if document_id is None or not str(document_id).strip():
                    warnings.append(f"Skipped legacy FAQ record {row_number}: missing id")
                    continue
                document_id = str(document_id).strip()
                if document_id in seen_ids:
                    continue
                normalized = {
                    **record,
                    "document_id": document_id,
                    "category": record.get("category", "FAQ"),
                    "source_file": fallback_faq_file.name,
                    "title": _title(record),
                    "primary_question": record.get("primary_question") or record.get("question"),
                    "searchable_text": _searchable_text(record),
                    "original_record": record,
                }
                normalized["source_reference"] = {
                    "document_id": document_id,
                    "category": normalized["category"],
                    "source_file": fallback_faq_file.name,
                    "title": normalized["title"],
                }
                documents.append(normalized)
                seen_ids.add(document_id)
            if documents:
                files_loaded.append(fallback_faq_file.name)
        except (OSError, json.JSONDecodeError) as exc:
            warnings.append(f"Could not read fallback FAQ dataset: {exc}")

    for warning in warnings:
        logger.warning(warning)
    logger.info("Loaded %d academic records from %d files", len(documents), len(files_loaded))
    return CorpusLoadResult(documents=documents, warnings=warnings, files_loaded=files_loaded)