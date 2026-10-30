from app.services.faq_service import FAQService
from app.services.retrieval_engine import AcademicRetrievalEngine
from app.services.text_preprocessing import normalize_text
from scripts.evaluate_retrieval import _score_method, deterministic_split


class StubSemanticRetriever:
    model_name = "test-model"
    model = object()
    load_error = None

    def __init__(self, document_id: str | None):
        self.document_id = document_id

    def search(self, query, records, top_k=3):
        if self.document_id is None:
            return []
        record = next(record for record in records if record["document_id"] == self.document_id)
        return [{"record": record, "score": 0.8, "method": "semantic"}]


class FixedSemanticRetriever:
    model_name = "test-model"
    model = object()
    load_error = None

    def __init__(self, hits):
        self.hits = hits

    def search(self, query, records, top_k=3):
        return self.hits[:top_k]


class FailingSemanticRetriever:
    model_name = "test-model"
    model = object()
    load_error = None

    def search(self, query, records, top_k=3):
        raise RuntimeError("model encoding failed")


def _records():
    return [
        {"document_id": "F1", "category": "Faculty", "name": "Dr Lee", "subjects": ["AI"], "searchable_text": "faculty teacher artificial intelligence"},
        {"document_id": "C1", "category": "Courses", "course": "AI Basics", "searchable_text": "course artificial intelligence credits"},
        {"document_id": "H1", "category": "Hostel", "rule": "quiet hours", "searchable_text": "hostel quiet hours residential rules"},
    ]


def test_tfidf_keeps_document_ids_and_returns_matching_record():
    records = _records()
    engine = AcademicRetrievalEngine(records, StubSemanticRetriever(None), tfidf_min_score=0.0)

    hits = engine.search("faculty teacher artificial intelligence", "tfidf")

    assert hits[0]["document_id"] == "F1"
    assert hits[0]["record"]["name"] == "Dr Lee"


def test_tfidf_vectorizer_fits_once_on_faq_questions():
    records = [
        {
            "document_id": "C1",
            "question": "How long is the B.E. programme at BEC?",
            "answer": "The B.E. programme lasts four years.",
            "searchable_text": "unrelated answer text should not enter the index",
        },
        {
            "document_id": "C2",
            "question": "What is the MBA programme duration?",
            "answer": "The MBA programme lasts two years.",
        },
    ]
    service = FAQService(records=records)

    assert service._vectorizer is not None
    assert service._matrix is not None
    assert service._matrix.shape[0] == 2
    assert "bec" in service._vectorizer.vocabulary_
    assert "unrelated" not in service._vectorizer.vocabulary_
    assert "four" not in service._vectorizer.vocabulary_
    assert service._record_indices == [0, 1]


def test_tfidf_exact_case_and_punctuation_variants_return_original_record():
    record = {
        "document_id": "C1",
        "question": "How long is the B.E. programme at BEC?",
        "answer": "The B.E. programme lasts four years.",
        "category": "Courses",
        "source_url": "https://example.test/course",
    }
    service = FAQService(records=[record])

    hits = service.search_tfidf("HOW LONG IS THE B.E. PROGRAMME AT BEC!!!")

    assert hits[0]["record"] is record
    assert hits[0]["score"] == 1.0
    assert hits[0]["record"]["answer"] == "The B.E. programme lasts four years."
    assert hits[0]["record"]["category"] == "Courses"
    assert hits[0]["record"]["source_url"] == "https://example.test/course"


def test_tfidf_ranks_exact_question_before_a_shared_term_candidate():
    records = [
        {
            "document_id": "C1",
            "question": "How long is the B.E. programme at BEC?",
            "answer": "Four years.",
        },
        {
            "document_id": "C2",
            "question": "Which B.E. programmes are offered at BEC?",
            "answer": "The college offers several programmes.",
        },
    ]
    service = FAQService(records=records)

    hits = service.search_tfidf("How long is the B.E. programme at BEC?", top_k=2, threshold=0)

    assert [hit["record"]["document_id"] for hit in hits] == ["C1", "C2"]
    assert hits[0]["score"] > hits[1]["score"]


def test_tfidf_returns_stable_original_order_for_tied_matches():
    records = [
        {"document_id": "first", "question": "Where is the BEC library?"},
        {"document_id": "second", "question": "Where is the BEC library?"},
    ]
    service = FAQService(records=records)

    first_results = service.search_tfidf("Where is the BEC library?", top_k=2, threshold=0)
    second_results = service.search_tfidf("Where is the BEC library?", top_k=2, threshold=0)

    assert [hit["record"]["document_id"] for hit in first_results] == ["first", "second"]
    assert [hit["record"]["document_id"] for hit in second_results] == ["first", "second"]


def test_tfidf_handles_empty_unknown_and_empty_vocabulary_cases():
    service = FAQService(records=[{"document_id": "C1", "question": "How long is B.E.?"}])

    assert service.search_tfidf("") == []
    assert service.search_tfidf(" \t ") == []
    assert service.search_tfidf("unrecognizedzebraquantum") == []
    assert service.search_tfidf("anything", top_k=0) == []

    no_questions = FAQService(records=[
        {"document_id": "blank", "question": " !!! "},
        {"document_id": "answer-only", "answer": "Do not index this answer."},
    ])
    assert no_questions._vectorizer is None
    assert no_questions._matrix is None
    assert no_questions.search_tfidf("answer") == []


def test_tfidf_common_paraphrase_returns_relevant_candidate():
    record = {
        "document_id": "C1",
        "question": "How long is the B.E. programme at BEC?",
        "answer": "The B.E. programme lasts four years.",
    }
    service = FAQService(records=[record])

    hits = service.search_tfidf("What is the duration of the B.E. degree?", threshold=0.01)

    assert hits
    assert hits[0]["record"] is record


def test_tfidf_category_scope_reuses_full_index_without_rebuilding():
    records = [
        {"document_id": "C1", "category": "Courses", "question": "How long is B.E.?"},
        {"document_id": "F1", "category": "Fees", "question": "How much is the B.E. fee?"},
    ]
    engine = AcademicRetrievalEngine(records, StubSemanticRetriever(None), tfidf_min_score=0.0)
    index = engine.tfidf
    category_records = [records[1]]

    hits = engine.search(
        "How much is the B.E. fee?",
        mode="tfidf",
        records=category_records,
    )

    assert engine.tfidf is index
    assert hits
    assert all(hit["record"] is records[1] for hit in hits)


def test_category_scope_excludes_semantic_candidates_outside_category():
    records = [
        {"document_id": "C1", "category": "Courses", "question": "How long is the B.E. programme?"},
        {"document_id": "F1", "category": "Fees", "question": "What is the B.E. fee?"},
    ]
    semantic = FixedSemanticRetriever([
        {"record": records[0], "score": 0.9},
        {"record": records[1], "score": 0.8},
    ])
    engine = AcademicRetrievalEngine(records, semantic, tfidf_min_score=0.0, semantic_min_score=0.0)

    hits = engine.search(
        "What is the B.E. fee?",
        mode="hybrid",
        records=[records[1]],
    )

    assert hits
    assert all(hit["record"]["category"] == "Fees" for hit in hits)
    assert all(hit["document_id"] == "F1" for hit in hits)


def test_hybrid_fuses_duplicate_candidates_once_and_ranks_shared_hit_first():
    records = [
        {"document_id": "C1", "category": "Courses", "question": "How long is the B.E. programme?"},
        {"document_id": "C2", "category": "Courses", "question": "Which B.E. programmes are offered?"},
    ]
    semantic = FixedSemanticRetriever([
        {"record": records[0], "score": 0.9},
        {"record": records[1], "score": 0.8},
    ])
    engine = AcademicRetrievalEngine(records, semantic, tfidf_min_score=0.0, semantic_min_score=0.0)

    hits = engine.search("How long is the B.E. programme?", mode="hybrid", top_k=2)

    ids = [hit["document_id"] for hit in hits]
    assert len(ids) == len(set(ids)) == 2
    assert ids[0] == "C1"
    assert all(hit["method"] == "hybrid_rrf" for hit in hits)


def test_hybrid_keeps_distinct_records_that_have_no_optional_ids():
    records = [
        {"category": "Courses", "question": "Where is the BEC library?"},
        {"category": "Courses", "question": "Where is the BEC library?"},
    ]
    semantic = FixedSemanticRetriever([
        {"record": records[0], "score": 0.9},
        {"record": records[1], "score": 0.8},
    ])
    engine = AcademicRetrievalEngine(records, semantic, tfidf_min_score=0.0, semantic_min_score=0.0)

    hits = engine.search("Where is the BEC library?", mode="hybrid", top_k=2)

    assert len(hits) == 2
    assert {id(hit["record"]) for hit in hits} == {id(record) for record in records}
    assert all(hit["document_id"] == "" for hit in hits)


def test_semantic_exception_falls_back_to_tfidf_results():
    records = [
        {"document_id": "H1", "category": "Hostel", "question": "What hostel quiet hours are required?"},
    ]
    engine = AcademicRetrievalEngine(records, FailingSemanticRetriever(), tfidf_min_score=0.0)

    hits = engine.search("What hostel quiet hours are required?", mode="semantic")

    assert hits
    assert hits[0]["document_id"] == "H1"
    assert hits[0]["method"] == "tfidf_fallback"


def test_hybrid_semantic_failure_still_returns_tfidf_results():
    records = [
        {"document_id": "H1", "category": "Hostel", "question": "What hostel quiet hours are required?"},
    ]
    engine = AcademicRetrievalEngine(records, FailingSemanticRetriever(), tfidf_min_score=0.0)

    hits = engine.search("What hostel quiet hours are required?", mode="hybrid")

    assert hits
    assert hits[0]["document_id"] == "H1"
    assert hits[0]["method"] == "tfidf_fallback"


def test_engine_ignores_semantic_hits_with_invalid_scores():
    records = [{"document_id": "H1", "category": "Hostel", "question": "What hostel quiet hours are required?"}]
    semantic = FixedSemanticRetriever([
        {"record": records[0], "score": float("nan")},
        {"record": records[0], "score": "invalid"},
    ])
    engine = AcademicRetrievalEngine(records, semantic, tfidf_min_score=0.0, semantic_min_score=0.0)

    hits = engine.search("What hostel quiet hours are required?", mode="semantic")

    assert hits
    assert hits[0]["method"] == "tfidf_fallback"


def test_semantic_retriever_handles_model_encoding_failure():
    from app.services.retrieval import SemanticRetriever

    class BrokenModel:
        def encode(self, *args, **kwargs):
            raise RuntimeError("encoding unavailable")

    retriever = SemanticRetriever()
    retriever.model = BrokenModel()
    retriever._load_attempted = True
    records = [{"document_id": "H1", "searchable_text": "hostel quiet hours"}]

    assert retriever.search("hostel quiet hours", records) == []
    assert retriever.model is None
    assert "encoding unavailable" in retriever.load_error


def test_semantic_retriever_uses_stable_dataset_order_for_tied_scores():
    from app.services.retrieval import SemanticRetriever

    class EqualScoreModel:
        def encode(self, texts, **kwargs):
            import numpy as np

            if len(texts) == 1:
                return np.array([[1.0, 0.0]])
            return np.array([[1.0, 0.0] for _ in texts])

    retriever = SemanticRetriever()
    retriever.model = EqualScoreModel()
    retriever._load_attempted = True
    records = [
        {"document_id": "first", "searchable_text": "first question"},
        {"document_id": "second", "searchable_text": "second question"},
    ]

    hits = retriever.search("any query", records, top_k=2)

    assert [hit["record"]["document_id"] for hit in hits] == ["first", "second"]


def test_tfidf_pipeline_uses_shared_normalization_and_returns_original_record():
    record = {
        "document_id": "FEE-1",
        "category": "Fees",
        "question": "Is the ₹25,000 fee refundable?",
        "answer": "The fee is not refundable.",
        "verification_note": "User-provided; unverified.",
        "source_url": "https://example.test/fees",
        "searchable_text": "Is the ₹25,000 fee refundable? The fee is not refundable. Fees",
    }
    engine = AcademicRetrievalEngine(
        [record],
        StubSemanticRetriever(None),
        tfidf_min_score=0.0,
    )

    hits = engine.search("  IS the ₹ 25,000 fee NOT refundable?!  ", mode="tfidf")

    assert normalize_text(" B.E.\t2026-27 ") == "b e 2026 27"
    assert hits[0]["record"] is record
    assert hits[0]["record"]["answer"] == "The fee is not refundable."
    assert hits[0]["record"]["verification_note"] == "User-provided; unverified."
    assert hits[0]["record"]["source_url"] == "https://example.test/fees"


def test_tfidf_pipeline_returns_no_hits_for_empty_or_invalid_queries():
    engine = AcademicRetrievalEngine(_records(), StubSemanticRetriever(None), tfidf_min_score=0.0)

    assert engine.search("", mode="tfidf") == []
    assert engine.search(" \t ", mode="tfidf") == []
    assert engine.search(None, mode="tfidf") == []


def test_semantic_mode_falls_back_to_tfidf_when_model_unavailable():
    records = _records()
    engine = AcademicRetrievalEngine(records, StubSemanticRetriever(None), tfidf_min_score=0.0)

    hits = engine.search("hostel quiet hours", "semantic")

    assert hits[0]["document_id"] == "H1"
    assert hits[0]["method"] == "tfidf_fallback"


def test_semantic_mode_uses_relevant_faq_answer():
    records = [
        {
            "document_id": "FAQ-1",
            "category": "FAQ",
            "answer": "Submit the application through the admissions portal.",
            "searchable_text": "question: how do I apply for admission answer: submit the application through the admissions portal",
        },
        {"document_id": "HOSTEL-1", "category": "Hostel", "searchable_text": "hostel quiet hours"},
    ]
    engine = AcademicRetrievalEngine(records, StubSemanticRetriever("FAQ-1"), tfidf_min_score=0.0)

    hits = engine.search("How can I apply for admission?", mode="semantic")

    assert hits[0]["document_id"] == "FAQ-1"
    assert hits[0]["record"]["answer"] == "Submit the application through the admissions portal."



def test_hybrid_fuses_rankings_and_preserves_ids():
    records = _records()
    engine = AcademicRetrievalEngine(records, StubSemanticRetriever("C1"), tfidf_min_score=0.0)

    hits = engine.search("faculty artificial intelligence", "hybrid")

    assert hits
    assert all(hit["document_id"] in {record["document_id"] for record in records} for hit in hits)
    assert all(hit["method"] == "hybrid_rrf" for hit in hits)


def test_deterministic_split_is_repeatable_and_disjoint():
    rows = [{"query_id": str(index), "query": f"question {index}", "document_id": f"D{index % 3}"} for index in range(20)]

    first_train, first_validation = deterministic_split(rows, seed=42)
    second_train, second_validation = deterministic_split(rows, seed=42)

    assert first_train == second_train
    assert first_validation == second_validation
    assert not ({row["query_id"] for row in first_train} & {row["query_id"] for row in first_validation})
    assert len(first_validation) == 4


def test_evaluation_metrics_measure_top1_top3_and_reciprocal_rank():
    class RankedEngine:
        def search(self, query, mode, top_k, apply_threshold):
            return [
                {"document_id": "A"},
                {"document_id": "B"},
                {"document_id": "C"},
            ]

    metrics = _score_method(
        RankedEngine(),
        [
            {"query": "first query", "document_id": "B"},
            {"query": "second query", "document_id": "missing"},
        ],
        "tfidf",
    )

    assert metrics["top1_accuracy"] == 0.0
    assert metrics["top3_recall"] == 0.5
    assert metrics["mrr"] == 0.25