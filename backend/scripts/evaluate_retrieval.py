import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.services.answerability import is_answerable_hit
from app.services.data_loader import load_bec_corpus
from app.services.retrieval import SemanticRetriever
from app.services.retrieval_engine import AcademicRetrievalEngine

EVALUATION_QUERIES_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "retrieval_evaluation_queries.json"
)
REPORT_DIRECTORY = Path(__file__).resolve().parents[1] / "reports"
MIN_CATEGORY_CASES = 2
TOP_K = 3


def load_evaluation_queries(
    path: Path,
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read evaluation queries from {path}: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("queries"), list):
        raise ValueError(f"{path} must contain a JSON object with a 'queries' array")

    records_by_id = {
        str(record.get("document_id")): record
        for record in records
        if record.get("document_id") is not None
    }
    if len(records_by_id) != len(records):
        raise ValueError("FAQ records must have unique, non-empty document IDs for evaluation")

    seen_query_ids: set[str] = set()
    seen_paraphrases: set[str] = set()
    normalized_questions = {
        str(record.get("question", "")).strip().casefold()
        for record in records
        if isinstance(record.get("question"), str)
    }
    queries = payload["queries"]
    if not queries:
        raise ValueError("Evaluation query set must not be empty")

    for row_number, query in enumerate(queries, start=1):
        if not isinstance(query, dict):
            raise ValueError(f"Evaluation query {row_number} must be a JSON object")
        query_id = query.get("query_id")
        query_text = query.get("query")
        query_type = query.get("query_type")
        if not isinstance(query_id, str) or not query_id.strip() or query_id in seen_query_ids:
            raise ValueError(f"Evaluation query {row_number} has a missing or duplicate query_id")
        if not isinstance(query_text, str):
            raise ValueError(f"Evaluation query {query_id!r} must have a string query")
        if query_type not in {"original", "paraphrase", "unsupported", "empty"}:
            raise ValueError(f"Evaluation query {query_id!r} has an unsupported query_type")
        seen_query_ids.add(query_id)

        if query_type in {"original", "paraphrase"}:
            expected_id = query.get("expected_document_id")
            record = records_by_id.get(str(expected_id))
            if record is None:
                raise ValueError(
                    f"Evaluation query {query_id!r} references unknown FAQ ID {expected_id!r}"
                )
            if query.get("category") != record.get("category"):
                raise ValueError(f"Evaluation query {query_id!r} has a category inconsistent with its FAQ")
            canonical_question = record.get("question")
            normalized_query = query_text.strip().casefold()
            if query_type == "original" and query_text != canonical_question:
                raise ValueError(f"Original evaluation query {query_id!r} must equal its FAQ question")
            if query_type == "paraphrase":
                if not query_text.strip() or normalized_query in normalized_questions:
                    raise ValueError(
                        f"Paraphrase query {query_id!r} must be non-empty and distinct from FAQ questions"
                    )
                if normalized_query in seen_paraphrases:
                    raise ValueError(f"Duplicate paraphrase text in query {query_id!r}")
                seen_paraphrases.add(normalized_query)
            if query.get("expected_no_result", False):
                raise ValueError(f"Answerable query {query_id!r} cannot expect a rejection")
        else:
            if query.get("expected_no_result") is not True:
                raise ValueError(f"Rejection query {query_id!r} must set expected_no_result to true")
            if query.get("expected_document_id") is not None:
                raise ValueError(f"Rejection query {query_id!r} cannot have an expected FAQ ID")
            if query_type == "empty" and query_text.strip():
                raise ValueError(f"Empty query {query_id!r} must contain only whitespace or be empty")
            if query_type == "unsupported" and not query_text.strip():
                raise ValueError(f"Unsupported query {query_id!r} must contain non-empty text")
    return queries


def _query_result(
    engine: AcademicRetrievalEngine,
    query: dict[str, Any],
    method: str,
) -> dict[str, Any]:
    raw_hits = engine.search(query["query"], mode=method, top_k=12, apply_threshold=True)
    hits = [
        hit
        for hit in raw_hits
        if is_answerable_hit(
            query["query"],
            hit,
            method,
            tfidf_min_score=settings.answer_tfidf_min_score,
            semantic_min_score=settings.answer_semantic_min_score,
        )
    ]
    raw_ranked_ids = [str(hit.get("document_id", "")) for hit in raw_hits[:TOP_K]]
    ranked_ids = [str(hit.get("document_id", "")) for hit in hits[:TOP_K]]
    query_type = query["query_type"]
    is_answerable = query_type in {"original", "paraphrase"}
    expected_id = str(query["expected_document_id"]) if is_answerable else None
    rank = ranked_ids.index(expected_id) + 1 if expected_id in ranked_ids else None
    raw_rank = raw_ranked_ids.index(expected_id) + 1 if expected_id in raw_ranked_ids else None
    rejected = not ranked_ids
    return {
        "query_id": query["query_id"],
        "query": query["query"],
        "query_type": query_type,
        "category": query.get("category"),
        "expected_document_id": expected_id,
        "retrieved_document_ids": ranked_ids,
        "expected_rank": rank,
        "expected_rank_before_answerability": raw_rank,
        "top1_correct": rank == 1 if is_answerable else None,
        "top3_correct": rank is not None and rank <= TOP_K if is_answerable else None,
        "reciprocal_rank": 1 / rank if rank is not None else 0.0,
        "top1_correct_before_answerability": raw_rank == 1 if is_answerable else None,
        "top3_correct_before_answerability": raw_rank is not None and raw_rank <= TOP_K if is_answerable else None,
        "reciprocal_rank_before_answerability": 1 / raw_rank if raw_rank is not None else 0.0,
        "retrieved_before_answerability": raw_ranked_ids,
        "rejected": rejected,
        "rejected_before_answerability": not raw_hits,
        "rejection_correct": rejected if not is_answerable else None,
        "rejection_correct_before_answerability": (
            not raw_hits if not is_answerable else None
        ),
        "answerable": is_answerable,
        "method": hits[0].get("method") if hits else None,
    }


def _metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    answerable = [result for result in results if result["answerable"]]
    rejection = [result for result in results if not result["answerable"]]
    answerable_count = len(answerable)
    rejection_count = len(rejection)
    return {
        "answerable_queries": answerable_count,
        "rejection_queries": rejection_count,
        "top1_accuracy": (
            sum(bool(result["top1_correct"]) for result in answerable) / answerable_count
            if answerable_count else 0.0
        ),
        "top1_accuracy_before_answerability": (
            sum(bool(result.get("top1_correct_before_answerability")) for result in answerable) / answerable_count
            if answerable_count else 0.0
        ),
        "accepted_answerable_top1_accuracy": (
            sum(
                bool(result["top1_correct"])
                for result in answerable
                if not result["rejected"]
            )
            / sum(not result["rejected"] for result in answerable)
            if any(not result["rejected"] for result in answerable) else 0.0
        ),
        "top3_accuracy": (
            sum(bool(result["top3_correct"]) for result in answerable) / answerable_count
            if answerable_count else 0.0
        ),
        "top3_accuracy_before_answerability": (
            sum(bool(result.get("top3_correct_before_answerability")) for result in answerable) / answerable_count
            if answerable_count else 0.0
        ),
        "mrr": (
            sum(result["reciprocal_rank"] for result in answerable) / answerable_count
            if answerable_count else 0.0
        ),
        "mrr_before_answerability": (
            sum(result.get("reciprocal_rank_before_answerability", 0.0) for result in answerable) / answerable_count
            if answerable_count else 0.0
        ),
        "accepted_answerable_queries": sum(not result["rejected"] for result in answerable),
        "answerable_acceptance_rate": (
            sum(not result["rejected"] for result in answerable) / answerable_count
            if answerable_count else 0.0
        ),
        "answerable_no_result_count": sum(result["rejected"] for result in answerable),
        "correct_rejections": sum(bool(result["rejection_correct"]) for result in rejection),
        "false_acceptances": sum(not result["rejected"] for result in rejection),
        "correct_rejections_before_answerability": sum(
            bool(result.get("rejection_correct_before_answerability"))
            for result in rejection
        ),
        "false_acceptances_before_answerability": sum(
            not result.get("rejected_before_answerability", False)
            for result in rejection
        ),
        "rejection_accuracy_before_answerability": (
            sum(
                bool(result.get("rejection_correct_before_answerability"))
                for result in rejection
            ) / rejection_count
            if rejection_count else 0.0
        ),
        "rejection_accuracy": (
            sum(bool(result["rejection_correct"]) for result in rejection) / rejection_count
            if rejection_count else 0.0
        ),
    }


def evaluate_method(
    engine: AcademicRetrievalEngine,
    queries: list[dict[str, Any]],
    method: str,
) -> dict[str, Any]:
    results = [_query_result(engine, query, method) for query in queries]
    metrics = _metrics(results)
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        if result["answerable"]:
            by_category[result["category"]].append(result)
    category_metrics = {
        category: _metrics(category_results)
        for category, category_results in sorted(by_category.items())
        if len(category_results) >= MIN_CATEGORY_CASES
    }
    return {
        "status": "measured",
        "metrics": metrics,
        "category_metrics": category_metrics,
        "results": results,
    }


def deterministic_split(
    rows: list[dict[str, str]],
    seed: int = 42,
    validation_fraction: float = 0.2,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    validation_count = min(len(rows), max(1, round(len(rows) * validation_fraction))) if rows else 0
    validation_indices = set(indices[:validation_count])
    train = [row for index, row in enumerate(rows) if index not in validation_indices]
    validation = [row for index, row in enumerate(rows) if index in validation_indices]
    return train, validation


def _score_method(
    engine: AcademicRetrievalEngine,
    rows: list[dict[str, str]],
    method: str,
) -> dict[str, Any]:
    results = []
    for row in rows:
        hits = engine.search(row["query"], mode=method, top_k=TOP_K, apply_threshold=False)
        ranked_ids = [str(hit.get("document_id", "")) for hit in hits]
        expected = row.get("document_id", row.get("expected_document_id", ""))
        rank = ranked_ids.index(expected) + 1 if expected in ranked_ids else None
        results.append({
            "answerable": True,
            "top1_correct": rank == 1,
            "top3_correct": rank is not None and rank <= TOP_K,
            "reciprocal_rank": 1 / rank if rank is not None else 0.0,
        })
    count = len(results)
    return {
        "status": "measured",
        "validation_queries": count,
        "top1_accuracy": sum(result["top1_correct"] for result in results) / count if count else 0.0,
        "top3_recall": sum(result["top3_correct"] for result in results) / count if count else 0.0,
        "mrr": sum(result["reciprocal_rank"] for result in results) / count if count else 0.0,
    }


def _percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def _markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# CampusConnect AI retrieval evaluation",
        "",
        f"- FAQ records loaded: {report['knowledge_documents']}",
        f"- Answerable test queries: {report['answerable_query_count']}",
        f"- Rejection test queries: {report['rejection_query_count']}",
        "- Retrieval scores use the production retrieval engine and configured relevance thresholds.",
        "",
        "## Overall results",
        "",
        "| Mode | Status | Top-1 before -> after | Top-3 before -> after | MRR before -> after | Answerable accepted | Accepted top-1 | Rejection accuracy before -> after | False accepts before -> after |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method, result in report["metrics"].items():
        if "metrics" not in result:
            lines.append(f"| {method} | {result['status']} | N/A | N/A | N/A | N/A | N/A | N/A | N/A |")
            continue
        metrics = result["metrics"]
        lines.append(
            f"| {method} | {result['status']} | {_percent(metrics['top1_accuracy_before_answerability'])} -> {_percent(metrics['top1_accuracy'])} "
            f"| {_percent(metrics['top3_accuracy_before_answerability'])} -> {_percent(metrics['top3_accuracy'])} "
            f"| {_percent(metrics['mrr_before_answerability'])} -> {_percent(metrics['mrr'])} "
            f"| {_percent(metrics['answerable_acceptance_rate'])} "
            f"| {_percent(metrics['accepted_answerable_top1_accuracy'])} "
            f"| {_percent(metrics['rejection_accuracy_before_answerability'])} -> {_percent(metrics['rejection_accuracy'])} "
            f"| {metrics['false_acceptances_before_answerability']} -> {metrics['false_acceptances']} |"
        )
    lines.extend(["", "## Category results", ""])
    for method, result in report["metrics"].items():
        if not result.get("category_metrics"):
            continue
        lines.extend([f"### {method}", "", "| Category | Queries | Top-1 | Top-3 | MRR |", "|---|---:|---:|---:|---:|"])
        for category, metrics in result["category_metrics"].items():
            lines.append(
                f"| {category} | {metrics['answerable_queries']} "
                f"| {_percent(metrics['top1_accuracy'])} | {_percent(metrics['top3_accuracy'])} "
                f"| {_percent(metrics['mrr'])} |"
            )
        lines.append("")

    examples = report["examples"]
    lines.extend(["## Example retrievals (hybrid mode)", ""])
    for label, key in (("Successful", "successful"), ("Incorrect or low-ranked", "incorrect_or_low_ranked")):
        lines.extend([f"### {label}", ""])
        if not examples.get(key):
            lines.extend(["No examples in this group.", ""])
            continue
        for result in examples[key]:
            expected = result["expected_document_id"]
            lines.append(
                f"- `{result['query_id']}` ({result['query_type']}, {result.get('category') or 'rejection'}): "
                f"{result['query']} — expected `{expected or 'no result'}`, "
                f"rank `{result['expected_rank']}`, retrieved {result['retrieved_document_ids']}."
            )
        lines.append("")
    lines.extend(["### Rejection queries", ""])
    for result in examples["rejections"]:
        outcome = "correctly rejected" if result["rejection_correct"] else "false acceptance"
        lines.append(
            f"- `{result['query_id']}`: {result['query']} — {outcome}; "
            f"retrieved {result['retrieved_document_ids']}."
        )
    lines.append("")
    semantic_reason = report["metrics"]["semantic"].get("reason")
    if semantic_reason:
        lines.extend([f"Semantic mode could not run: {semantic_reason}", ""])
    lines.extend(["## Interpretation and limitations", ""])
    lines.extend(f"- {note}" for note in report["limitations"])
    lines.append("")
    return "\n".join(lines)


def run_evaluation() -> dict[str, Any]:
    loaded = load_bec_corpus(settings.bec_dataset_path)
    queries = load_evaluation_queries(EVALUATION_QUERIES_PATH, loaded.documents)
    semantic = SemanticRetriever(settings.semantic_model)
    engine = AcademicRetrievalEngine(
        loaded.documents,
        semantic,
        tfidf_min_score=settings.tfidf_min_score,
        semantic_min_score=settings.semantic_min_score,
        rrf_constant=settings.hybrid_rrf_constant,
    )

    metrics = {
        method: evaluate_method(engine, queries, method)
        for method in ("tfidf", "semantic", "hybrid")
    }
    semantic_available = semantic.model is not None
    if not semantic_available:
        metrics["semantic"] = {
            "status": "unavailable",
            "reason": semantic.load_error or "Semantic model could not be initialized.",
        }
        metrics["hybrid"]["status"] = "measured_with_tfidf_fallback"

    hybrid_results = metrics["hybrid"].get("results", [])
    successful = sorted([
        result for result in hybrid_results
        if result["answerable"] and result["expected_rank"] == 1
    ], key=lambda result: (result["query_type"] != "paraphrase", result["query_id"]))[:3]
    incorrect = [
        result for result in hybrid_results
        if result["answerable"] and (result["expected_rank"] is None or result["expected_rank"] > 1)
    ][:3]
    report = {
        "dataset": "bec_campusconnect_dataset.json",
        "evaluation_queries_file": EVALUATION_QUERIES_PATH.name,
        "knowledge_documents": len(loaded.documents),
        "answerable_query_count": sum(query["query_type"] in {"original", "paraphrase"} for query in queries),
        "rejection_query_count": sum(query["query_type"] in {"unsupported", "empty"} for query in queries),
        "retrieval_mode": "production AcademicRetrievalEngine",
        "semantic_model": settings.semantic_model,
        "semantic_model_available": semantic_available,
        "metrics": metrics,
        "examples": {
            "successful": successful,
            "incorrect_or_low_ranked": incorrect,
            "rejections": [
                result for result in hybrid_results
                if not result["answerable"]
            ],
        },
        "limitations": [
            "This is a small, manually authored benchmark, not a representative sample of real student questions.",
            "Paraphrase quality and expected FAQ labels were authored by a person; results do not prove real-world accuracy.",
            "The original-question subset is naturally close to the indexed FAQ wording and should not be read as paraphrase performance.",
            "The rejection benchmark is small and manually authored, so its rejection accuracy is not a guarantee for unseen unsupported questions.",
            "The source FAQ records were not modified; missing source IDs use the loader's stable BEC-ROW-NNN IDs.",
            "Expected document IDs are consulted only after retrieval to score rankings; they are never passed into search.",
        ],
    }
    REPORT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    json_path = REPORT_DIRECTORY / "retrieval_evaluation.json"
    markdown_path = REPORT_DIRECTORY / "retrieval_evaluation.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    markdown_path.write_text(_markdown_report(report), encoding="utf-8")
    print(_markdown_report(report))
    print(f"JSON report saved to {json_path}")
    print(f"Markdown report saved to {markdown_path}")
    return report


if __name__ == "__main__":
    run_evaluation()
