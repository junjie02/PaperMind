import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root, then fall back to current directory
_env_paths = [
    Path(__file__).resolve().parent.parent.parent / ".env",  # PaperMind/.env
    Path(".env"),
]
for p in _env_paths:
    if p.exists():
        load_dotenv(p)
        break


@dataclass
class Config:
    # Optional LLM client (OpenAI / DeepSeek compatible). Not currently consumed
    # by the default Claude Code pipeline, kept as a hook for future use.
    llm_api_key: str = field(default_factory=lambda: os.getenv("DEEPSEEK_API_KEY", ""))
    llm_base_url: str = field(default_factory=lambda: os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"))
    llm_model: str = field(default_factory=lambda: os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro"))
    llm_max_tokens: int = field(default_factory=lambda: int(os.getenv("DEEPSEEK_MAX_TOKENS", "4096")))
    llm_json_mode: bool = field(default_factory=lambda: os.getenv("DEEPSEEK_JSON_MODE", "1") == "1")

    http_proxy: str = field(default_factory=lambda: os.getenv("HTTP_PROXY", ""))
    search_timeout: int = field(default_factory=lambda: int(os.getenv("SEARCH_TIMEOUT", "900")))
    worker_model: str = field(default_factory=lambda: os.getenv("WORKER_MODEL", ""))
    pdf_max_attempts: int = field(default_factory=lambda: int(os.getenv("PDF_MAX_ATTEMPTS", "3")))
    pdf_retry_sleep: float = field(default_factory=lambda: float(os.getenv("PDF_RETRY_SLEEP", "2.0")))
    pdf_timeout: float = field(default_factory=lambda: float(os.getenv("PDF_TIMEOUT", "60")))

    default_target: int = 30
    workers_per_round: int = 1
    db_path: str = ""
    pdf_dir: str = ""
