import json
from pathlib import Path

import pytest

from app.services.data_loader import load_academic_corpus, load_bec_corpus


def test_loader_preserves_fields_and_source_metadata(tmp_path):
    knowledge_base = tmp_path / "knowledge_base"
    knowledge_base.mkdir()
    (knowledge_base / "faculty.json").write_text(
        json.dumps([{"document_id": "F-1", "name": "A. Student", "subjects": ["AI"]}]),
        encoding="utf-8",
    )

    result = load_academic_corpus(knowledge_base)

    assert len(result.documents) == 1
    document = result.documents[0]
    assert document["document_id"] == "F-1"
    assert document["category"] == "Faculty"
    assert document["source_file"] == "faculty.json"
    assert document["original_record"]["subjects"] == ["AI"]
    assert "subjects: AI" in document["searchable_text"]
    assert document["source_reference"]["document_id"] == "F-1"
    assert any("Dataset file missing" in warning for warning in result.warnings)


def test_loader_reports_malformed_and_duplicate_records(tmp_path):
    knowledge_base = tmp_path / "knowledge_base"
    knowledge_base.mkdir()
    (knowledge_base / "courses.json").write_text(
        json.dumps([
            {"document_id": "C-1", "course": "Data Science"},
            {"document_id": "C-1", "course": "Duplicate"},
            {"course": "No ID"},
            "not a record",
        ]),
        encoding="utf-8",
    )
    (knowledge_base / "faq.json").write_text("{broken", encoding="utf-8")

    result = load_academic_corpus(knowledge_base)

    assert [record["document_id"] for record in result.documents] == ["C-1"]
    assert any("duplicate document_id" in warning for warning in result.warnings)
    assert any("missing document_id" in warning for warning in result.warnings)
    assert any("non-object record" in warning for warning in result.warnings)
    assert any("Could not read faq.json" in warning for warning in result.warnings)


def test_loader_skips_records_without_searchable_fields(tmp_path):
    knowledge_base = tmp_path / "knowledge_base"
    knowledge_base.mkdir()
    (knowledge_base / "faculty.json").write_text(
        json.dumps([{"document_id": "EMPTY"}]),
        encoding="utf-8",
    )

    result = load_academic_corpus(knowledge_base)

    assert result.documents == []
    assert any("empty searchable record 'EMPTY'" in warning for warning in result.warnings)


def test_bec_loader_reads_project_dataset_and_generates_stable_ids():
    dataset_path = Path(__file__).resolve().parents[1] / "data" / "bec_campusconnect_dataset.json"

    result = load_bec_corpus(dataset_path)

    assert len(result.documents) == result.records_read == 170
    assert result.invalid_records == 0
    assert result.dataset_metadata["dataset_name"] == "CampusConnect AI - Basaveshwar Engineering College Knowledge Dataset"
    assert len({record["document_id"] for record in result.documents}) == 170
    assert sum(record["document_id"].startswith("BEC-ROW-") for record in result.documents) == 71
    assert result.documents[0]["category"] == "Admissions"
    assert result.documents[0]["source_url"].startswith("https://")
    assert "not_independently_verified" in result.documents[0]["verification_status"]
    assert result.documents[0]["answer"] in result.documents[0]["searchable_text"]
    assert result.documents[0]["original_record"]["question"] == result.documents[0]["question"]
    assert any("Generated local IDs for 71 records" in warning for warning in result.warnings)


def test_bec_loader_validates_required_fields_and_preserves_optional_metadata(tmp_path):
    dataset_path = tmp_path / "bec.json"
    records = [
        {
            "question": " Where is BEC? ",
            "answer": "  Bagalkote.  ",
            "category": "Admissions",
            "source_url": "https://example.test/bec",
            "source_status": "not_independently_verified",
            "verification_note": "Check with the college.",
        },
        {"question": "Missing answer", "category": "Admissions"},
        {"question": "Missing category", "answer": "No category."},
        {"question": "   ", "answer": "Blank question.", "category": "Admissions"},
        {"question": "Wrong answer type", "answer": None, "category": "Admissions"},
        {"question": "Wrong category type", "answer": "Answer.", "category": 123},
    ]
    dataset_path.write_text(json.dumps({"dataset_name": "Test BEC", "records": records}), encoding="utf-8")

    result = load_bec_corpus(dataset_path)

    assert len(result.documents) == 1
    assert result.documents[0]["document_id"] == "BEC-ROW-001"
    assert result.documents[0]["question"] == records[0]["question"]
    assert result.documents[0]["answer"] == records[0]["answer"]
    assert result.documents[0]["category"] == records[0]["category"]
    assert result.documents[0]["institution"] == "Basaveshwar Engineering College, Bagalkote"
    assert result.documents[0]["source_reference"]["url"] == "https://example.test/bec"
    assert result.documents[0]["source_reference"]["verification_note"] == "Check with the college."
    assert result.documents[0]["original_record"] == records[0]
    assert result.invalid_records == 5
    assert any("missing or invalid answer" in warning for warning in result.warnings)
    assert any("missing or invalid category" in warning for warning in result.warnings)
    assert any("missing or invalid question" in warning for warning in result.warnings)


def test_bec_loader_uses_backend_relative_path_independent_of_current_directory(monkeypatch, tmp_path):
    dataset_path = Path(__file__).resolve().parents[1] / "data" / "bec_campusconnect_dataset.json"
    monkeypatch.chdir(tmp_path)

    result = load_bec_corpus(Path("data") / dataset_path.name)

    assert result.records_read == 170
    assert len(result.documents) == 170


def test_bec_loader_reports_missing_and_invalid_dataset_files(tmp_path):
    missing_path = tmp_path / "missing.json"
    with pytest.raises(FileNotFoundError, match="BEC chatbot dataset not found"):
        load_bec_corpus(missing_path)

    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text("{invalid", encoding="utf-8")
    with pytest.raises(ValueError, match="Could not read BEC chatbot dataset"):
        load_bec_corpus(invalid_path)


def test_bec_loader_rejects_empty_dataset(tmp_path):
    dataset_path = tmp_path / "empty.json"
    dataset_path.write_text(json.dumps({"records": []}), encoding="utf-8")

    with pytest.raises(ValueError, match="contains no valid question-and-answer records"):
        load_bec_corpus(dataset_path)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ([], "must contain a JSON object"),
        ({"dataset_name": "BEC"}, "missing its 'records' array"),
        ({"records": [None]}, "contains no valid question-and-answer records"),
    ],
)
def test_bec_loader_rejects_invalid_dataset_structures(tmp_path, payload, message):
    dataset_path = tmp_path / "bec.json"
    dataset_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_bec_corpus(dataset_path)