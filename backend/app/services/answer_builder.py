from typing import Any


_HIDDEN_FIELDS = {
    "document_id",
    "id",
    "source_file",
    "source_url",
    "url",
    "verification_status",
    "is_demo",
}


def _display_value(value: Any) -> str:
    if isinstance(value, dict):
        return ", ".join(f"{key.replace('_', ' ')}: {_display_value(item)}" for key, item in value.items())
    if isinstance(value, (list, tuple, set)):
        return ", ".join(_display_value(item) for item in value)
    if value is None:
        return ""
    return str(value).strip()


def build_answer(record: dict[str, Any]) -> str:
    saved_answer = record.get("answer")
    if isinstance(saved_answer, str) and saved_answer.strip():
        return saved_answer.strip()

    label = record.get("title") or "Academic record"
    normalized_label = str(label).strip().casefold()
    fields = []
    seen_values: set[str] = set()
    original = record.get("original_record", record)
    for key, value in original.items():
        if key in _HIDDEN_FIELDS or key == "question" or value is None:
            continue
        text = _display_value(value)
        normalized_value = text.casefold()
        if text and normalized_value != normalized_label and normalized_value not in seen_values:
            fields.append(f"{key.replace('_', ' ').capitalize()}: {text}")
            seen_values.add(normalized_value)
    if not fields:
        return str(label)
    return f"{label}: " + ". ".join(fields) + "."