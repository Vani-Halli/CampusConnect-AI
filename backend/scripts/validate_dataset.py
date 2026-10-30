import json
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.services.data_loader import EXPECTED_FILES, load_academic_corpus


def _records_from_payload(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("records", "documents", "data", "items"):
            nested = payload.get(key)
            if isinstance(nested, (list, dict)):
                return _records_from_payload(nested)
        return [payload]
    return []


def validate_dataset() -> dict[str, Any]:
    dataset_dir = settings.academic_dataset_dir / "knowledge_base"
    corpus = load_academic_corpus(dataset_dir)
    loaded_by_file: dict[str, int] = {}
    for document in corpus.documents:
        source_file = str(document.get("source_file", ""))
        loaded_by_file[source_file] = loaded_by_file.get(source_file, 0) + 1

    files = []
    missing_files = []
    total_invalid_or_duplicate = 0
    for file_name in EXPECTED_FILES:
        path = dataset_dir / file_name
        if not path.is_file():
            missing_files.append(file_name)
            files.append({
                "file": file_name,
                "json_records": 0,
                "searchable_documents": 0,
                "invalid_or_duplicate_records": 0,
                "status": "missing",
            })
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            total_invalid_or_duplicate += 1
            files.append({
                "file": file_name,
                "json_records": 0,
                "searchable_documents": 0,
                "invalid_or_duplicate_records": 1,
                "status": f"unreadable: {exc}",
            })
            continue

        records = _records_from_payload(payload)
        searchable = loaded_by_file.get(file_name, 0)
        invalid_or_duplicate = max(0, len(records) - searchable)
        total_invalid_or_duplicate += invalid_or_duplicate
        files.append({
            "file": file_name,
            "json_records": len(records),
            "searchable_documents": searchable,
            "invalid_or_duplicate_records": invalid_or_duplicate,
            "status": "loaded" if searchable else "no searchable records",
        })

    unique_courses = {
        str(document.get("original_record", {}).get("course", "")).strip().casefold()
        for document in corpus.documents
        if document.get("category") == "Courses"
        and str(document.get("original_record", {}).get("course", "")).strip()
    }
    return {
        "dataset_directory": str(dataset_dir),
        "files": files,
        "totals": {
            "json_records": sum(file["json_records"] for file in files),
            "searchable_documents": len(corpus.documents),
            "unique_courses": len(unique_courses),
            "invalid_or_duplicate_records": total_invalid_or_duplicate,
            "missing_files": len(missing_files),
        },
        "missing_file_names": missing_files,
        "loader_warnings": corpus.warnings,
    }


if __name__ == "__main__":
    print(json.dumps(validate_dataset(), indent=2))
