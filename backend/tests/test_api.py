import pytest
from fastapi.testclient import TestClient

from app import main as main_module
from app.core.config import settings
from app.main import app

client = TestClient(app)
BEC_INSTITUTION = "Basaveshwar Engineering College, Bagalkote"


def test_health_reports_loaded_bec_dataset():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["institution"] == BEC_INSTITUTION
    assert data["dataset_name"] == "CampusConnect AI - Basaveshwar Engineering College Knowledge Dataset"
    assert data["corpus_documents"] == 170
    assert data["dataset_records_read"] == 170
    assert data["dataset_invalid_records"] == 0
    assert data["dataset_generated_ids"] == 71


def test_cors_allows_private_lan_frontend_origin_in_development():
    response = client.options(
        "/api/health",
        headers={
            "Origin": "http://10.41.156.210:5173",
            "Access-Control-Request-Method": "GET",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://10.41.156.210:5173"


def test_cors_allows_configured_local_vite_origins():
    for origin in ("http://localhost:5173", "http://127.0.0.1:5173"):
        response = client.options(
            "/api/chat",
            headers={
                "Origin": origin,
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )

        assert response.status_code == 200
        assert response.headers["access-control-allow-origin"] == origin


def test_cors_does_not_allow_an_unconfigured_origin():
    response = client.options(
        "/api/chat",
        headers={
            "Origin": "https://unconfigured.example.invalid",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )

    assert response.status_code == 400
    assert "access-control-allow-origin" not in response.headers
    assert "*" not in settings.allowed_origins


def test_faqs_loaded_from_bec_dataset():
    response = client.get("/api/faqs")
    assert response.status_code == 200
    records = response.json()
    assert len(records) == 170
    assert all(record["institution"] == BEC_INSTITUTION for record in records)
    assert all(record["document_id"] and record["answer"] for record in records)
    assert all(record["source_file"] == "bec_campusconnect_dataset.json" for record in records)


def test_categories_and_counts_match_dataset():
    categories = client.get("/api/categories").json()
    counts = client.get("/api/category-counts").json()

    assert set(counts) == {
        "Admissions", "Courses", "Fees", "Placements", "Student Life", "Examinations",
        "Academic Verification", "Administration", "College Contacts", "Scholarships", "Student Support",
    }
    assert sum(counts.values()) == 170
    assert counts["Courses"] == 8
    assert counts["Placements"] == 30
    assert "Faculty" not in categories
    assert "Attendance" not in categories


def test_bec_course_question_returns_record_answer_and_source_metadata():
    response = client.post(
        "/api/chat",
        json={"question": "How long is the B.E. programme at BEC?", "mode": "tfidf"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "answered"
    assert body["category"] == "Courses"
    assert "4 years" in body["answer"]
    assert body["sources"][0]["url"].startswith("https://")
    assert body["sources"][0]["source_file"] == "bec_campusconnect_dataset.json"
    assert "not_independently_verified" in body["sources"][0]["verification_status"]
    assert body["sources"][0]["matched_question"] == "How long is the B.E. programme at BEC?"
    assert "verification_note" in body["sources"][0]
    assert body["retrieved_document_ids"]
    assert "Additional relevant information:" not in body["answer"]
    assert body["follow_up_questions"]
    assert body["intent"] == "specific"
    assert body["retrieval_method"]
    assert body["generation_status"]


def test_broad_course_query_returns_paginated_dataset_records():
    response = client.post(
        "/api/chat",
        json={"question": "What courses are offered?", "mode": "tfidf", "page_size": 3},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == "collection"
    assert body["category"] == "Courses"
    assert body["total"] == 8
    assert len(body["records"]) == 3
    assert body["has_more"]
    assert "B.E." in body["answer"]
    assert "M.Tech" in body["answer"]
    assert "Ph.D." in body["answer"]
    assert "MBA" in body["answer"]
    assert "MCA" in body["answer"]
    assert body["follow_up_questions"]
    assert all(record["answer"] for record in body["records"])
    assert all(source["url"].startswith("https://") for source in body["sources"])
    assert all(source["verification_status"] for source in body["sources"])
    assert all("verification_note" in source for source in body["sources"])


def test_contextual_follow_up_uses_previous_question_without_changing_visible_query():
    response = client.post(
        "/api/chat",
        json={
            "question": "How long is it?",
            "history": ["How long is the B.E. programme at BEC?"],
            "mode": "tfidf",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "answered"
    assert body["query"] == "How long is it?"
    assert body["category"] == "Courses"
    assert "4 years" in body["answer"]
    assert body["sources"][0]["matched_question"] == "How long is the B.E. programme at BEC?"


def test_count_explicitly_describes_faq_records_not_real_world_entities():
    response = client.post(
        "/api/chat",
        json={"question": "How many Courses records are available?", "mode": "tfidf"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == "count"
    assert body["total"] == 8
    assert "FAQ records" in body["answer"]
    assert "not a count of courses" in body["answer"]


def test_count_of_real_world_courses_is_not_answered_with_record_count():
    response = client.post(
        "/api/chat",
        json={"question": "How many courses are offered at BEC?", "mode": "tfidf"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["intent"] != "count"
    assert "8 courses" not in body["answer"].casefold()


def test_placements_question_uses_matching_bec_record():
    response = client.post(
        "/api/chat",
        json={"question": "How many students were eligible for placement in the 2025 batch at BEC?", "mode": "tfidf"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "answered"
    assert body["category"] == "Placements"
    assert "494 eligible students" in body["answer"]
    assert body["sources"][0]["url"]


def test_be_branch_paraphrase_returns_discipline_records_with_sources():
    response = client.post(
        "/api/chat",
        json={"question": "Which B.E. branches are available?", "mode": "tfidf"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "answered"
    assert body["category"] == "Courses"
    assert body["retrieved_document_ids"][0] in {"BEC009", "BEC010"}
    assert "engineering" in body["answer"].casefold()
    assert "disciplines" in body["sources"][0]["matched_question"].casefold()
    assert body["sources"][0]["url"].startswith("https://")


def test_be_fee_question_with_admissions_word_uses_fee_record():
    response = client.post(
        "/api/chat",
        json={"question": "What is the fee structure for B.E. admissions?", "mode": "tfidf"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "answered"
    assert body["category"] == "Fees"
    assert body["retrieved_document_ids"][0] == "BEC014"
    assert "Exact numeric schedules were not included" in body["answer"]
    assert body["sources"][0]["matched_question"] == "What are the B.E. fee components for 2026-27?"


def test_general_contact_question_does_not_return_placement_contact():
    response = client.post(
        "/api/chat",
        json={"question": "How can I contact Basaveshwar Engineering College?", "mode": "tfidf"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "answered"
    assert body["category"] == "College Contacts"
    assert body["retrieved_document_ids"][0] in {"BEC-ROW-153", "BEC-ROW-164"}
    assert "becplacement@gmail.com" not in body["answer"]
    assert "becprincipal@yahoo.com" in body["answer"]
    assert body["sources"][0]["url"].startswith("https://")


def test_placement_opportunities_query_summarizes_matching_overview_records():
    response = client.post(
        "/api/chat",
        json={"question": "What placement opportunities are available?", "mode": "tfidf"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == "collection"
    assert body["category"] == "Placements"
    assert body["total"] < 30
    assert {"BEC042", "BEC043", "BEC044", "BEC046", "BEC047", "BEC049"}.issubset(
        set(body["retrieved_document_ids"])
    )
    answer = body["answer"]
    batch_2025_end = answer.index("\n2026\n")
    year_2026_end = answer.index("\nOther figures")
    assert "2025 batch\n- 494 eligible students" in answer
    assert "- 250 students placed for the 2025 batch" in answer
    assert "- 442 placement offers for the 2025 batch" in answer
    assert "450 placement offers for 2026" in answer[batch_2025_end:year_2026_end]
    assert "450 placement offers for 2026" not in answer[:batch_2025_end]
    assert "Average package: ₹4.0 LPA" in answer[year_2026_end:]
    assert "does not clearly specify the batch or reporting period" in answer
    assert "130+ recruiters" in answer[year_2026_end:]
    assert "Training and Placement Office" not in answer
    assert all(source["url"].startswith("https://") for source in body["sources"])
    assert body["follow_up_questions"]


def test_placement_policy_summary_includes_tpo_only_when_requested():
    response = client.post(
        "/api/chat",
        json={"question": "List placement policy information", "mode": "tfidf"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == "collection"
    assert "Placement policy\n- Campus recruitment is organized in priority slots based on CTC bands." in body["answer"]
    assert "Training and Placement Office manages the process" in body["answer"]
    assert body["sources"]


def test_generic_fee_structure_question_asks_which_program():
    response = client.post(
        "/api/chat",
        json={"question": "What is the fee structure?", "mode": "tfidf"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "unsupported"
    assert body["category"] == "Fees"
    assert body["sources"] == []
    assert "B.E., MBA, or MCA" in body["answer"]
    assert "look up the fee information available" in body["answer"]


def test_admission_process_does_not_misrepresent_eligibility_as_procedure():
    response = client.post(
        "/api/chat",
        json={"question": "What is the B.E. admission process?", "mode": "tfidf"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "answered"
    assert body["retrieved_document_ids"][0] == "BEC013"
    assert "entrance test" in body["answer"].casefold()
    assert "complete step-by-step admission process" in body["answer"]
    assert "45%" not in body["answer"]


def test_internship_question_uses_explicit_dataset_gap_fallback():
    response = client.post(
        "/api/chat",
        json={"question": "Are internships available at BEC?", "mode": "tfidf"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "unsupported"
    assert body["category"] == "Placements"
    assert body["sources"] == []
    assert "does not include information about internships" in body["answer"]


def test_missing_faculty_and_attendance_topics_are_not_fabricated():
    for question, category in (
        ("Who teaches computer science at BEC?", "Faculty"),
        ("What is the BEC attendance policy?", "Attendance"),
    ):
        response = client.post("/api/chat", json={"question": question, "mode": "tfidf"})

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "unsupported"
        assert body["category"] == category
        assert body["sources"] == []
        assert "not available in the current BEC knowledge base" in body["answer"]
        assert "appropriate college office" in body["answer"]


def test_irrelevant_question_uses_safe_bec_fallback():
    response = client.post(
        "/api/chat",
        json={"question": "What is quantum archaeology on Jupiter?", "mode": "tfidf"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "unsupported"
    assert body["sources"] == []
    assert body["retrieved_document_ids"] == []
    assert "not available in the current BEC knowledge base" in body["answer"]
    assert "rephrase" in body["answer"]
    assert body["generation_status"] == "not_attempted"


def test_off_topic_questions_with_strong_lexical_overlap_do_not_return_faq_answers():
    for question in (
        "Does BEC have a campus observatory that lends telescopes?",
        "What are the campus aquarium fish-feeding hours?",
    ):
        response = client.post("/api/chat", json={"question": question, "mode": "tfidf"})

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "unsupported"
        assert body["sources"] == []
        assert body["retrieved_document_ids"] == []
        assert "rephrase" in body["answer"]


def test_supported_hostel_paraphrase_returns_the_original_faq_answer_and_source():
    response = client.post(
        "/api/chat",
        json={"question": "Are student hostels available at BEC?", "mode": "tfidf"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "answered"
    assert body["category"] == "Student Life"
    assert body["sources"]
    assert body["sources"][0]["matched_question"]
    assert body["sources"][0]["verification_status"]


def test_person_question_does_not_get_an_answer_from_a_documents_faq():
    response = client.post(
        "/api/chat",
        json={
            "question": "Who is eligible to ask BEC to verify academic documents?",
            "mode": "tfidf",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "unsupported"
    assert body["sources"] == []
    assert "rephrase" in body["answer"]


def test_whitespace_only_chat_question_is_rejected():
    response = client.post("/api/chat", json={"question": "   ", "mode": "tfidf"})

    assert response.status_code == 400
    assert response.json()["detail"] == "Question cannot be empty."


def test_knowledge_search_returns_bec_source_metadata():
    response = client.get(
        "/api/knowledge",
        params={"query": "B.E. programme duration", "category": "Courses", "mode": "tfidf"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["items"]
    assert body["items"][0]["category"] == "Courses"
    assert body["items"][0]["source_file"] == "bec_campusconnect_dataset.json"
    assert body["items"][0]["source_reference"]["url"].startswith("https://")


def test_retrieval_evaluation_endpoint_serves_saved_bec_metrics():
    response = client.get("/api/evaluation")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "measured"
    assert body["dataset"] == "bec_campusconnect_dataset.json"
    assert body["knowledge_documents"] == 170
    assert body["answerable_query_count"] == 33
    assert body["rejection_query_count"] == 8
    assert body["metrics"]["tfidf"]["status"] == "measured"
    assert body["metrics"]["tfidf"]["top1_accuracy"] == pytest.approx(0.8788, abs=0.0001)
    assert body["metrics"]["tfidf"]["rejection_accuracy"] == 1.0


def test_retrieval_evaluation_endpoint_reports_missing_saved_report(monkeypatch, tmp_path):
    monkeypatch.setattr(main_module, "EVALUATION_REPORT_PATH", tmp_path / "missing.json")

    response = client.get("/api/evaluation")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "not_run"
    assert "No saved BEC retrieval evaluation report" in body["message"]


def test_invalid_pdf_signature_is_rejected(monkeypatch):
    monkeypatch.setattr(settings, "admin_api_key", "test-admin-key")
    response = client.post(
        "/api/documents/upload",
        files={"file": ("not-a-pdf.pdf", b"not a pdf", "application/pdf")},
        headers={"X-Admin-Key": "test-admin-key"},
    )
    assert response.status_code == 400
    assert "valid PDF signature" in response.json()["detail"]


def test_deleting_unknown_document_returns_404(monkeypatch):
    monkeypatch.setattr(settings, "admin_api_key", "test-admin-key")
    response = client.delete("/api/documents/999999", headers={"X-Admin-Key": "test-admin-key"})
    assert response.status_code == 404


def test_document_management_requires_admin_key(monkeypatch):
    monkeypatch.setattr(settings, "admin_api_key", "test-admin-key")
    response = client.get("/api/documents")
    assert response.status_code == 403


def test_document_management_is_disabled_without_admin_key(monkeypatch):
    monkeypatch.setattr(settings, "admin_api_key", "")
    response = client.get("/api/documents")
    assert response.status_code == 503


def test_invalid_chat_question():
    response = client.post("/api/chat", json={"question": ""})
    assert response.status_code == 422


def test_feedback_persistence():
    response = client.post(
        "/api/feedback",
        json={"question": "Test question", "answer": "Test answer", "helpful": True, "category": "Admissions"},
    )
    assert response.status_code == 200
    assert response.json()["status"] == "saved"


def test_analytics_summary():
    response = client.get("/api/analytics/summary")
    assert response.status_code == 200
    assert "total_questions" in response.json()
