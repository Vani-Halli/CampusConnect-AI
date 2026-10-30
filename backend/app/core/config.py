from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings

BASE_DIR = Path(__file__).resolve().parents[3]
BACKEND_DIR = Path(__file__).resolve().parents[2]
DEFAULT_ACADEMIC_DATASET_DIR = BASE_DIR / "backend" / "data" / "academic_retrieval"
DEFAULT_BEC_DATASET_PATH = BACKEND_DIR / "data" / "bec_campusconnect_dataset.json"


class Settings(BaseSettings):
    app_name: str = "CampusConnect AI"
    app_env: str = "development"
    debug: bool = True
    backend_host: str = "0.0.0.0"
    backend_port: int = 8000
    database_url: str = "sqlite:///./campusconnect.db"
    allowed_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    upload_dir: str = "./uploads"
    max_upload_size_mb: int = 10
    admin_api_key: str = ""
    retrieval_mode: str = "semantic"
    llm_provider: str = "none"
    llm_api_key: str = ""
    llm_model: str = ""
    llm_base_url: str = ""
    semantic_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    academic_dataset_dir: Path = DEFAULT_ACADEMIC_DATASET_DIR
    bec_dataset_path: Path = DEFAULT_BEC_DATASET_PATH
    semantic_min_score: float = 0.20
    tfidf_min_score: float = 0.08
    answer_tfidf_min_score: float = 0.12
    answer_semantic_min_score: float = 0.20
    hybrid_rrf_constant: int = 60

    class Config:
        env_file = BASE_DIR / ".env"
        extra = "ignore"


settings = Settings()
