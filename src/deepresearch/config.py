import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root, then fall back to current directory
_env_paths = [
    Path(__file__).resolve().parent.parent.parent / ".env",  # DeepResearch/.env
    Path(".env"),
]
for p in _env_paths:
    if p.exists():
        load_dotenv(p)
        break


@dataclass
class Config:
    llm_api_key: str = field(default_factory=lambda: os.getenv("DEEPSEEK_API_KEY", ""))
    llm_base_url: str = field(default_factory=lambda: os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"))
    llm_model: str = field(default_factory=lambda: os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro"))
    llm_max_tokens: int = field(default_factory=lambda: int(os.getenv("DEEPSEEK_MAX_TOKENS", "4096")))
    llm_json_mode: bool = field(default_factory=lambda: os.getenv("DEEPSEEK_JSON_MODE", "1") == "1")
    http_proxy: str = field(default_factory=lambda: os.getenv("HTTP_PROXY", ""))
    search_backend: str = field(default_factory=lambda: os.getenv("SEARCH_BACKEND", "claude"))
    search_timeout: int = field(default_factory=lambda: int(os.getenv("SEARCH_TIMEOUT", "900")))
    arxiv_timeout: float = field(default_factory=lambda: float(os.getenv("ARXIV_TIMEOUT", "10")))
    arxiv_retries: int = field(default_factory=lambda: int(os.getenv("ARXIV_RETRIES", "2")))

    default_target: int = 30
    workers_per_round: int = 4
    max_rounds: int = 20
    stall_rounds_limit: int = 1
    db_path: str = "papers.db"
