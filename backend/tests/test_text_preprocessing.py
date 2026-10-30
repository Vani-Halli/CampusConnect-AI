from pathlib import Path

import pytest

from app.services.data_loader import load_bec_corpus
from app.services.faq_service import FAQService
from app.services.text_preprocessing import normalize_text, tokenize_text


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("  BEC   ADMISSIONS  ", "bec admissions"),
        ("How\tmany\ncourses   are offered?", "how many courses are offered"),
        ("Fees: B.E. / M.Tech!", "fees b e m tech"),
        ("Fee ₹1,50,000.50 for 2026-27", "fee inr 150000.50 for 2026 27"),
        ("Course code CSE-101 and VTU", "course code cse 101 and vtu"),
        ("It is NOT allowed; cannot apply.", "it is not allowed cannot apply"),
        ("Café — विद्यापीठ", "café विद्यापीठ"),
        ("B.E. 2026-27, ₹ 25,000", "b e 2026 27 inr 25000"),
    ],
)
def test_normalize_text_handles_case_spacing_punctuation_and_unicode(raw, expected):
    assert normalize_text(raw) == expected


@pytest.mark.parametrize("value", ["", " \t\r\n ", None, 123, object()])
def test_normalize_text_handles_empty_and_invalid_inputs(value):
    assert normalize_text(value) == ""
    assert tokenize_text(value) == []


def test_tokenizer_preserves_numbers_dates_codes_abbreviations_and_negations():
    tokens = tokenize_text("B.E. CSE-101 2026-27 ₹25,000.50 not cannot")

    assert tokens == ["b", "e", "cse", "101", "2026", "27", "inr", "25000.50", "not", "cannot"]


def test_tfidf_matches_case_spacing_punctuation_and_unicode_variants():
    records = [{
        "document_id": "BEC-1",
        "question": "What is the B.E. programme duration?",
        "answer": "The B.E. programme duration is 4 years.",
        "category": "Courses",
        "searchable_text": "What is the B.E. programme duration? The B.E. programme duration is 4 years. Courses",
    }]
    service = FAQService(records=records)

    for question in (
        "WHAT IS THE B.E. PROGRAMME DURATION???",
        "  what   is the B.E. programme duration ",
        "What is the B E programme duration",
    ):
        hits = service.search_tfidf(question, threshold=0.01)
        assert hits
        assert hits[0]["record"] is records[0]


def test_tfidf_keeps_negations_and_fee_tokens_distinguishable():
    records = [
        {
            "document_id": "FEE-1",
            "question": "Is the ₹25,000 fee not refundable?",
            "answer": "The supplied information says the fee is not refundable.",
            "category": "Fees",
            "searchable_text": "Is the ₹25,000 fee not refundable? The supplied information says the fee is not refundable. Fees",
        },
        {
            "document_id": "FEE-2",
            "question": "Is the ₹25,000 fee refundable?",
            "answer": "The supplied information says the fee is refundable.",
            "category": "Fees",
            "searchable_text": "Is the ₹25,000 fee refundable? The supplied information says the fee is refundable. Fees",
        },
    ]
    service = FAQService(records=records)

    negative_hits = service.search_tfidf("Is the ₹25,000 fee not refundable?", threshold=0)
    positive_hits = service.search_tfidf("Is the ₹25,000 fee refundable?", threshold=0)

    assert negative_hits[0]["record"]["document_id"] == "FEE-1"
    assert positive_hits[0]["record"]["document_id"] == "FEE-2"


def test_bec_loader_metadata_and_original_answer_survive_retrieval():
    dataset_path = Path(__file__).resolve().parents[1] / "data" / "bec_campusconnect_dataset.json"
    loaded = load_bec_corpus(dataset_path)
    record = next(item for item in loaded.documents if item["document_id"] == "BEC001")
    service = FAQService(records=[record])

    hits = service.search_tfidf(record["question"].swapcase(), threshold=0.01)

    assert hits
    retrieved = hits[0]["record"]
    assert retrieved["answer"] == record["original_record"]["answer"]
    assert retrieved["category"] == record["original_record"]["category"]
    assert retrieved["source_url"] == record["original_record"]["source_url"]
    assert retrieved["source_reference"]["verification_note"] == record["original_record"]["verification_note"]
    assert retrieved["original_record"] == record["original_record"]
