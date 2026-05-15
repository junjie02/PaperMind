"""DeepResearch - Automated literature survey agent."""

from deepresearch.models import PaperRecord, WorkerTask
from deepresearch.orchestrator import Orchestrator
from deepresearch.config import Config

__all__ = [
    "PaperRecord",
    "WorkerTask",
    "Orchestrator",
    "Config",
]
