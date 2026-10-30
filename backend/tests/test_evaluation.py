import copy
import json

import pytest

from app.core.config import settings
from app.services.data_loader import load_bec_corpus
from scripts.evaluate_retrieval import (
    EVALUATION_QUERIES_PATH,
    _metrics,
    evaluate_method,
    load_evaluation_queries,
)


class RankedEngine:
    def __init__(self, ranked_by_query):
        self.ranked_by_query = ranked_by_query
        self.calls = []

    def search(self, query, mode, top_k, apply_threshold):
        self.calls.append((query, mode, top_k, apply_threshold))
        return [
            {
                "document_id": document_id,
                "score": 0.9,
                "method": "tfidf",
                "record": {"question": query, "answer": ""},
            }
            for document_id in self.ranked_by_query.get(query, [])[:top_k]
        ]


@pytest.fixture
def benchmark_records():
    return [
        {
            "document_id": "BEC001",
            "question": "Which B.E. course does BEC offer?",
            "category": "Admissions",
        },
        {
            "document_id": "BEC-ROW-002",
            "question": "Where is the library?",
            "category": "Student Life",
        },
    ]


@pytest.fixture
def benchmark_queries():
    return [
        {
            "query_id": "original-one",
            "query_type": "original",
            "query": "Which B.E. course does BEC offer?",
            "expected_document_id": "BEC001",
            "category": "Admissions",
        },
        {
            "query_id": "paraphrase-two",
            "query_type": "paraphrase",
            "query": "Where can I find the library?",
            "expected_document_id": "BEC-ROW-002",
            "category": "Student Life",
        },
        {
            "query_id": "unsupported",
            "query_type": "unsupported",
            "query": "What are telescope borrowing hours?",
            "expected_no_result": True,
        },
        {
            "query_id": "empty",
            "query_type": "empty",
            "query": "",
            "expected_no_result": True,
        },
    ]


def test_metric_calculation_for_known_rankings_and_rejections():
    results = [
        {
            "answerable": True,
            "top1_correct": False,
            "top3_correct": True,
            "reciprocal_rank": 0.5,
            "rejected": False,
        },
        {
            "answerable": True,
            "top1_correct": False,
            "top3_correct": False,
            "reciprocal_rank": 0.0,
            "rejected": True,
        },
        {
            "answerable": False,
            "rejection_correct": True,
            "rejected": True,
            "rejection_correct_before_answerability": False,
            "rejected_before_answerability": False,
        },
        {
            "answerable": False,
            "rejection_correct": False,
            "rejected": False,
            "rejection_correct_before_answerability": True,
            "rejected_before_answerability": True,
        },
    ]

    metrics = _metrics(results)

    assert metrics["answerable_queries"] == 2
    assert metrics["rejection_queries"] == 2
    assert metrics["top1_accuracy"] == 0.0
    assert metrics["top3_accuracy"] == 0.5
    assert metrics["mrr"] == 0.25
    assert metrics["answerable_no_result_count"] == 1
    assert metrics["correct_rejections"] == 1
    assert metrics["false_acceptances"] == 1
    assert metrics["rejection_accuracy"] == 0.5
    assert metrics["rejection_accuracy_before_answerability"] == 0.5


def test_expected_ids_are_used_for_scoring_not_passed_into_retrieval(benchmark_queries):
    engine = RankedEngine({
        "Which B.E. course does BEC offer?": ["OTHER", "BEC001"],
        "Where can I find the library?": ["BEC-ROW-002"],
        "What are telescope borrowing hours?": [],
        "": [],
    })

    report = evaluate_method(engine, benchmark_queries, "tfidf")

    assert report["metrics"]["top1_accuracy"] == 0.5
    assert report["metrics"]["top3_accuracy"] == 1.0
    assert report["metrics"]["mrr"] == 0.75
    assert report["metrics"]["rejection_accuracy"] == 1.0
    assert all(len(call) == 4 for call in engine.calls)
    assert all(call[3] is True for call in engine.calls)


def test_evaluation_labels_include_original_paraphrase_and_rejections(
    tmp_path,
    benchmark_records,
    benchmark_queries,
):
    path = tmp_path / "queries.json"
    write_queries(path, benchmark_queries)
    validated = load_evaluation_queries(path, benchmark_records)
    assert [query["query_type"] for query in validated] == [
        "original",
        "paraphrase",
        "unsupported",
        "empty",
    ]


def test_invalid_expected_faq_id_is_rejected(tmp_path, benchmark_records, benchmark_queries):
    queries = copy.deepcopy(benchmark_queries)
    queries[1]["expected_document_id"] = "missing-id"
    path = tmp_path / "queries.json"
    write_queries(path, queries)

    with pytest.raises(ValueError, match="unknown FAQ ID"):
        load_evaluation_queries(path, benchmark_records)


def test_original_and_paraphrase_wording_are_validated(tmp_path, benchmark_records, benchmark_queries):
    queries = copy.deepcopy(benchmark_queries)
    queries[0]["query"] = "A modified original question"
    path = tmp_path / "queries.json"
    write_queries(path, queries)

    with pytest.raises(ValueError, match="must equal its FAQ question"):
        load_evaluation_queries(path, benchmark_records)

    queries = copy.deepcopy(benchmark_queries)
    queries[1]["query"] = "Where is the library?"
    write_queries(path, queries)
    with pytest.raises(ValueError, match="distinct from FAQ questions"):
        load_evaluation_queries(path, benchmark_records)


def test_production_benchmark_uses_loader_generated_ids_and_is_reproducible():
    records = load_bec_corpus(settings.bec_dataset_path).documents
    first_queries = load_evaluation_queries(EVALUATION_QUERIES_PATH, records)
    second_queries = load_evaluation_queries(EVALUATION_QUERIES_PATH, records)

    assert first_queries == second_queries
    assert any(query["expected_document_id"] == "BEC-ROW-101" for query in first_queries)
    assert len({query["query_id"] for query in first_queries}) == len(first_queries)


def test_benchmark_evaluation_is_reproducible(benchmark_queries):
    ranked = {
        "Which B.E. course does BEC offer?": ["BEC001"],
        "Where can I find the library?": ["BEC-ROW-002"],
        "What are telescope borrowing hours?": [],
        "": [],
    }

    first = evaluate_method(RankedEngine(ranked), benchmark_queries, "tfidf")
    second = evaluate_method(RankedEngine(ranked), benchmark_queries, "tfidf")

    assert first == second


def test_rejection_queries_must_not_have_expected_faq_ids(
    tmp_path,
    benchmark_records,
    benchmark_queries,
):
    queries = copy.deepcopy(benchmark_queries)
    queries[2]["expected_document_id"] = "BEC001"
    path = tmp_path / "queries.json"
    write_queries(path, queries)

    with pytest.raises(ValueError, match="cannot have an expected FAQ ID"):
        load_evaluation_queries(path, benchmark_records)


def write_queries(path, queries):
    path.write_text(json.dumps({"queries": queries}), encoding="utf-8")
