import os
from functools import lru_cache
from pathlib import Path

try:
    # Auto-load a project-root .env when present (dev convenience).
    # python-dotenv ships with uvicorn[standard]; skip silently if absent.
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


class Settings:
    def __init__(self) -> None:
        self.artifacts_dir = Path(os.getenv("ARTIFACTS_DIR", "artifacts"))
        self.phobert_repo = os.getenv("PHOBERT_REPO", "")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY") or None
        origins = os.getenv("CORS_ORIGINS", "http://localhost:3000")
        self.cors_origins = [o.strip() for o in origins.split(",") if o.strip()]
        self.max_text_len = int(os.getenv("MAX_TEXT_LEN", "5000"))
        self.max_batch_rows = int(os.getenv("MAX_BATCH_ROWS", "5000"))
        self.max_upload_bytes = int(os.getenv("MAX_UPLOAD_BYTES", "10_000_000".replace("_", "")))


@lru_cache
def get_settings() -> Settings:
    return Settings()
