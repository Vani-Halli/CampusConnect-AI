from app.services.answerability import has_topical_evidence, is_answerable_hit


def _hit(question, answer, score=0.4, method="tfidf", component_scores=None):
    hit = {
        "record": {"question": question, "answer": answer},
        "score": score,
        "method": method,
    }
    if component_scores is not None:
        hit["component_scores"] = component_scores
    return hit


def _answerable(query, hit, mode="tfidf"):
    return is_answerable_hit(
        query,
        hit,
        mode,
        tfidf_min_score=0.12,
        semantic_min_score=0.20,
    )


def test_relevant_exact_question_and_paraphrase_have_topical_evidence():
    record = {
        "question": "How long is the B.E. programme at BEC?",
        "answer": "The B.E. programme lasts four years.",
    }

    assert has_topical_evidence("How long is the B.E. programme at BEC?", record)
    assert has_topical_evidence("How many years does the B.E. course take?", record)


def test_unrelated_faq_is_rejected_even_when_tfidf_score_is_high():
    hit = _hit(
        "What are the office hours of Basaveshwar Engineering College?",
        "The office is open Monday to Saturday.",
        score=0.4054,
    )

    assert not _answerable("What are the campus aquarium fish-feeding hours?", hit)


def test_unsupported_campus_observatory_question_has_no_topical_match():
    hit = _hit(
        "How does BEC organize campus placement recruitment?",
        "The Training and Placement Office manages campus recruitment.",
        score=0.4182,
    )

    assert not _answerable("Does BEC have a campus observatory that lends telescopes?", hit)


def test_low_confidence_candidate_is_rejected():
    hit = _hit(
        "How long is the B.E. programme at BEC?",
        "The B.E. programme lasts four years.",
        score=0.119,
    )

    assert not _answerable("How many years does the B.E. course take?", hit)


def test_category_context_alone_does_not_make_a_candidate_relevant():
    hit = _hit(
        "Does BEC have a campus observatory?",
        "The college has an astronomy club.",
        score=0.9,
    )

    assert not _answerable("How much are the placement fees?", hit)


def test_person_question_does_not_use_a_document_requirements_faq_as_the_answer():
    hit = _hit(
        "Which documents are required for BEC academic verification?",
        "Applicants must provide the specified academic records.",
        score=0.1685,
    )

    assert not _answerable(
        "Who is eligible to ask BEC to verify academic documents?",
        hit,
    )


def test_score_thresholds_are_applied_in_the_correct_mode():
    tfidf_hit = _hit(
        "What is the B.E. programme duration?",
        "The programme takes four years.",
        score=0.13,
        method="tfidf",
    )
    semantic_hit = _hit(
        "What is the B.E. programme duration?",
        "The programme takes four years.",
        score=0.25,
        method="semantic",
    )
    hybrid_hit = _hit(
        "What is the B.E. programme duration?",
        "The programme takes four years.",
        score=0.02,
        method="hybrid_rrf",
        component_scores={"tfidf": 0.13, "semantic": 0.19},
    )

    assert _answerable("How long is B.E.?", tfidf_hit, "tfidf")
    assert _answerable("How long is B.E.?", semantic_hit, "semantic")
    assert _answerable("How long is B.E.?", hybrid_hit, "hybrid")
    assert not _answerable("How long is B.E.?", hybrid_hit, "semantic")


def test_semantic_mode_accepts_tfidf_fallback_using_tfidf_score_scale():
    hit = _hit(
        "How long is the B.E. programme at BEC?",
        "The B.E. programme lasts four years.",
        score=0.3,
        method="tfidf_fallback",
    )

    assert _answerable("How many years does B.E. take?", hit, "semantic")


def test_empty_query_is_never_answerable():
    hit = _hit("How long is B.E.?", "The programme takes four years.")

    assert not _answerable("", hit)
    assert not _answerable(" \t ", hit)
