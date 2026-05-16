import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

_env_paths = [
    Path(__file__).resolve().parent.parent.parent / ".env",
    Path(".env"),
]
for _p in _env_paths:
    if _p.exists():
        load_dotenv(_p)
        break


@dataclass
class Config:
    # LLM (OpenAI-compatible: MiniMax / DeepSeek / OpenAI)
    # Reads OPENAI_* first, falls back to legacy DEEPSEEK_* for compatibility
    llm_api_key: str = field(default_factory=lambda: (
        os.getenv("OPENAI_API_KEY") or os.getenv("DEEPSEEK_API_KEY", "")
    ))
    llm_base_url: str = field(default_factory=lambda: (
        os.getenv("OPENAI_BASE_URL") or os.getenv("DEEPSEEK_BASE_URL", "https://api.minimaxi.com/v1")
    ))
    llm_model: str = field(default_factory=lambda: (
        os.getenv("OPENAI_MODEL") or os.getenv("DEEPSEEK_MODEL", "MiniMax-M2.7")
    ))
    llm_max_tokens: int = field(default_factory=lambda: int(
        os.getenv("OPENAI_MAX_TOKENS") or os.getenv("DEEPSEEK_MAX_TOKENS", "4096")
    ))
    llm_json_mode: bool = field(default_factory=lambda: (
        os.getenv("OPENAI_JSON_MODE", os.getenv("DEEPSEEK_JSON_MODE", "1")) == "1"
    ))

    http_proxy: str = field(default_factory=lambda: os.getenv("HTTP_PROXY", ""))
    search_timeout: int = field(default_factory=lambda: int(os.getenv("SEARCH_TIMEOUT", "900")))
    worker_model: str = field(default_factory=lambda: os.getenv("WORKER_MODEL", ""))

    pdf_max_attempts: int = field(default_factory=lambda: int(os.getenv("PDF_MAX_ATTEMPTS", "3")))
    pdf_retry_sleep: float = field(default_factory=lambda: float(os.getenv("PDF_RETRY_SLEEP", "2.0")))
    pdf_timeout: float = field(default_factory=lambda: float(os.getenv("PDF_TIMEOUT", "60")))

    embedding_model: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))
    writer_model: str = field(default_factory=lambda: os.getenv("WRITER_MODEL", ""))
    write_timeout: int = field(default_factory=lambda: int(os.getenv("WRITE_TIMEOUT", "1800")))
    writer_concurrency: int = field(default_factory=lambda: int(os.getenv("WRITER_CONCURRENCY", "4")))
    writer_max_retries: int = field(default_factory=lambda: int(os.getenv("WRITER_MAX_RETRIES", "3")))

    papers_per_round: int = field(default_factory=lambda: int(os.getenv("PAPERS_PER_ROUND", "10")))

    # Sub-Agent concurrency and timeouts
    max_concurrent_agents: int = field(default_factory=lambda: int(os.getenv("MAX_CONCURRENT_AGENTS", "4")))
    agent_max_retries: int = field(default_factory=lambda: int(os.getenv("AGENT_MAX_RETRIES", "2")))
    explorer_timeout: int = field(default_factory=lambda: int(os.getenv("EXPLORER_TIMEOUT", "600")))
    researcher_timeout: int = field(default_factory=lambda: int(os.getenv("RESEARCHER_TIMEOUT", "900")))
    reviewer_timeout: int = field(default_factory=lambda: int(os.getenv("REVIEWER_TIMEOUT", "300")))
    polisher_timeout: int = field(default_factory=lambda: int(os.getenv("POLISHER_TIMEOUT", "300")))

    # Final review LLM (separate model for global polish)
    review_api_key: str = field(default_factory=lambda: os.getenv("REVIEW_API_KEY", ""))
    review_base_url: str = field(default_factory=lambda: os.getenv("REVIEW_BASE_URL", ""))
    review_model: str = field(default_factory=lambda: os.getenv("REVIEW_MODEL", ""))

    default_target: int = 30
    workers_per_round: int = 1
    db_path: str = ""
    pdf_dir: str = ""
