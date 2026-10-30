import csv
from pathlib import Path

from app.core.config import settings
from app.services.data_loader import load_academic_corpus
from app.services.retrieval import SemanticRetriever
from app.services.retrieval_engine import AcademicRetrievalEngine


def export_predictions() -> Path:
    dataset_dir = settings.academic_dataset_dir
    corpus = load_academic_corpus(dataset_dir / "knowledge_base")
    known_ids = {record["document_id"] for record in corpus.documents}
    if not known_ids:
        raise RuntimeError(f"No academic documents were loaded from {dataset_dir / 'knowledge_base'}")

    test_path = dataset_dir / "test_queries.csv"
    with test_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"query_id", "query"}.issubset(reader.fieldnames or []):
            raise ValueError(f"{test_path} must contain query_id and query columns")
        test_rows = list(reader)

    engine = AcademicRetrievalEngine(
        corpus.documents,
        SemanticRetriever(settings.semantic_model),
        tfidf_min_score=settings.tfidf_min_score,
        semantic_min_score=settings.semantic_min_score,
        rrf_constant=settings.hybrid_rrf_constant,
    )
    predictions = []
    for row in test_rows:
        hits = engine.search(row["query"], mode=settings.retrieval_mode, top_k=1, apply_threshold=False)
        if not hits:
            hits = engine.search(row["query"], mode="tfidf", top_k=1, apply_threshold=False)
        if not hits or hits[0]["document_id"] not in known_ids:
            raise ValueError(f"No valid document prediction for query_id {row['query_id']}")
        predictions.append({"query_id": row["query_id"], "document_id": hits[0]["document_id"]})

    if len(predictions) != len(test_rows):
        raise ValueError("Prediction count does not match the test query count")

    output_path = Path(__file__).resolve().parents[1] / "reports" / "submission.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["query_id", "document_id"])
        writer.writeheader()
        writer.writerows(predictions)
    print(f"Wrote {len(predictions)} predictions to {output_path}")
    return output_path


if __name__ == "__main__":
    export_predictions()