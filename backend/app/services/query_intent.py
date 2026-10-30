import re
from typing import Any


CATEGORY_ALIASES = {
    "Scholarships": ("scholarship", "scholarships", "financial assistance", "financial aid"),
    "Academic Verification": ("academic verification", "transcript", "marks card", "degree certificate", "verification"),
    "Faculty": ("faculty", "professor", "teacher", "teaches", "teaching", "lecturer"),
    "Attendance": ("attendance", "attendance policy", "attendance requirement"),
    "Examinations": ("examination", "examinations", "exam", "exams", "test", "tests"),
    "Fees": ("fee", "fees", "tuition", "payment"),
    "Admissions": ("admission", "admissions", "enrollment", "enrolment", "applicant", "application"),
    "Placements": ("placement", "placements", "career", "job", "recruitment", "internship", "internships", "ojt"),
    "Student Support": ("student support", "counseling", "counselling", "support service"),
    "Student Life": ("student life", "hostel", "campus life", "clubs", "events", "sports"),
    "Courses": ("course", "courses", "program", "programs", "programme", "programmes", "department", "branch"),
    "Administration": ("administration", "administrative", "principal", "office"),
    "College Contacts": ("college contact", "contact", "contact details", "contact number", "phone number", "email address"),
}

LIST_CUES = (
    "all ",
    "list ",
    "show ",
    "complete ",
    "every ",
    "browse ",
    "where can i find",
)


def classify_count_query(question: str) -> str | None:
    normalized = re.sub(r"\s+", " ", question.casefold()).strip()
    if not re.search(r"\b(?:how many|number of|total(?: number)? of|count of)\b", normalized):
        return None
    if not re.search(r"\b(?:records?|questions?|entries|faqs?)\b", normalized):
        return None
    return category_for_query(normalized)


def classify_collection_query(question: str) -> str | None:
    normalized = re.sub(r"\s+", " ", question.casefold()).strip()
    for category, aliases in CATEGORY_ALIASES.items():
        matched_category = any(alias in normalized for alias in aliases)
        explicit_list_request = any(cue in normalized for cue in LIST_CUES)
        broad_overview_request = bool(
            re.search(r"\bwhat\s+[\w -]+\s+(?:is|are|available)\b", normalized)
        )
        listing_request = explicit_list_request or broad_overview_request
        if category == "Placements" and broad_overview_request:
            listing_request = explicit_list_request or bool(
                re.search(r"\b(?:opportunit(?:y|ies)|overview|results?)\b", normalized)
            )
        if matched_category and listing_request:
            return category
    return None


def category_for_query(question: str) -> str | None:
    normalized = re.sub(r"\s+", " ", question.casefold()).strip()
    for category, aliases in CATEGORY_ALIASES.items():
        if any(alias in normalized for alias in aliases):
            return category
    return None


def filter_collection_records(
    question: str,
    category: str,
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str | None]:
    normalized = question.casefold()
    semester_match = re.search(r"\bsemester\s*(?:number\s*)?(\d+)\b", normalized)
    requested_date = re.search(r"\b(?:on|for)\s+(\d{1,2}\s+[a-z]+|\d{1,2}/\d{1,2}(?:/\d{2,4})?)\b", normalized)
    department_match = re.search(r"\bdepartment\s+(?:of\s+)?([a-z0-9-]+)\b", normalized)

    requested_field: str | None = None
    expected: str | None = None
    if semester_match:
        requested_field, expected = "semester", semester_match.group(1)
    elif requested_date:
        requested_field, expected = "date", requested_date.group(1)
    elif department_match:
        requested_field, expected = "department", department_match.group(1)

    if requested_field is None:
        return records, None

    if not any(requested_field in record.get("original_record", {}) for record in records):
        return [], f"The {category} records do not contain a '{requested_field}' field, so that filter cannot be applied."

    selected = []
    for record in records:
        value = record.get("original_record", {}).get(requested_field)
        if value is not None and expected.casefold() in str(value).casefold():
            selected.append(record)
    return selected, None


def unsupported_collection_fact(question: str, category: str, records: list[dict[str, Any]]) -> str | None:
    normalized = question.casefold()
    if category == "Placements" and re.search(r"\binternships?\b", normalized):
        internship_text = " ".join(
            " ".join(
                str(record.get(field) or "")
                for field in ("question", "primary_question", "title", "answer")
            ).casefold()
            for record in records
        )
        if not re.search(r"\binternships?\b", internship_text):
            return (
                "The current BEC knowledge base does not include information about internships. "
                "Please contact the appropriate college office to confirm."
            )

    searchable = " ".join(
        str(value).casefold()
        for record in records
        for value in record.get("original_record", {}).values()
    )
    year_match = re.search(r"\b(?:19|20)\d{2}\b", normalized)
    if year_match and year_match.group(0) not in searchable:
        return f"The current {category.lower()} records do not include information for {year_match.group(0)}."
    if any(term in normalized for term in ("package", "salary", "stipend")) and not any(
        term in searchable for term in ("package", "salary", "stipend")
    ):
        return f"The current {category.lower()} records do not contain placement compensation information."
    return None
