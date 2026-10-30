import re
from typing import Any

from app.services.text_preprocessing import normalize_text, tokenize_text

_NON_TOPICAL_TERMS = {
    "a", "about", "after", "all", "an", "and", "any", "are", "as", "at", "be",
    "been", "before", "being", "between", "can", "could", "did", "do", "does",
    "doing", "for", "from", "get", "give", "has", "have", "having", "how", "i",
    "in", "into", "is", "it", "its", "may", "me", "of", "on", "or", "please",
    "should", "tell", "that", "the", "their", "them", "there", "these", "this",
    "those", "to", "was", "we", "were", "what", "when", "where", "which", "who",
    "why", "will", "with", "would", "you", "your",
    "basaveshwar", "bec", "campus", "college", "engineering", "institution",
    "student", "students",
    "hour", "hours",
}
_TERM_ALIASES = {
    "course": "course",
    "courses": "course",
    "branch": "discipline",
    "branches": "discipline",
    "program": "course",
    "programme": "course",
    "programmes": "course",
    "duration": "long",
    "located": "location",
    "lead": "principal",
    "leads": "principal",
    "leader": "principal",
    "reach": "contact",
    "mention": "listed",
    "mentions": "listed",
    "eligible": "request",
    "ask": "request",
    "verify": "verification",
}


def _stem_simple(token: str) -> str:
    if len(token) > 5 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def _topical_terms(text: object) -> set[str]:
    tokens = tokenize_text(normalize_text(text))
    combined_tokens = []
    index = 0
    while index < len(tokens):
        if tokens[index:index + 2] == ["b", "e"]:
            combined_tokens.append("b_e")
            index += 2
        else:
            combined_tokens.append(tokens[index])
            index += 1
    return {
        _TERM_ALIASES.get(token, _stem_simple(token))
        for token in combined_tokens
        if token not in _NON_TOPICAL_TERMS and len(token) > 1
    }


def has_topical_evidence(query: str, record: dict[str, Any]) -> bool:
    record_question = (
        record.get("question")
        or record.get("primary_question")
        or record.get("title")
        or ""
    )
    if re.match(r"^\s*who\b", query, flags=re.IGNORECASE) and not re.match(
        r"^\s*who\b",
        str(record_question),
        flags=re.IGNORECASE,
    ):
        return False

    query_terms = _topical_terms(query)
    if not query_terms:
        return False

    record_text = " ".join(
        str(record.get(field) or "")
        for field in ("question", "primary_question", "title", "answer")
    )
    matching_terms = query_terms & _topical_terms(record_text)
    required_matches = min(2, len(query_terms))
    return len(matching_terms) >= required_matches


def is_answerable_hit(
    query: str,
    hit: dict[str, Any],
    mode: str,
    *,
    tfidf_min_score: float,
    semantic_min_score: float,
) -> bool:
    record = hit.get("record")
    if not isinstance(record, dict) or not normalize_text(query):
        return False

    method = hit.get("method")
    component_scores = hit.get("component_scores")
    if isinstance(component_scores, dict):
        tfidf_score = component_scores.get("tfidf")
        semantic_score = component_scores.get("semantic")
    elif method in {"tfidf", "tfidf_fallback"}:
        tfidf_score = hit.get("score")
        semantic_score = None
    elif method == "semantic":
        tfidf_score = None
        semantic_score = hit.get("score")
    else:
        tfidf_score = semantic_score = None

    if mode == "tfidf":
        sufficiently_scored = _meets_threshold(tfidf_score, tfidf_min_score)
    elif mode == "semantic":
        fallback_to_tfidf = method == "tfidf_fallback"
        threshold_score = tfidf_score if fallback_to_tfidf else semantic_score
        threshold = tfidf_min_score if fallback_to_tfidf else semantic_min_score
        sufficiently_scored = _meets_threshold(threshold_score, threshold)
    elif mode == "hybrid":
        sufficiently_scored = (
            _meets_threshold(tfidf_score, tfidf_min_score)
            or _meets_threshold(semantic_score, semantic_min_score)
        )
    else:
        sufficiently_scored = False
    return sufficiently_scored and has_topical_evidence(query, record)


def _meets_threshold(score: object, threshold: float) -> bool:
    try:
        numeric_score = float(score)
    except (TypeError, ValueError):
        return False
    return numeric_score >= threshold
