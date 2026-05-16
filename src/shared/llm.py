"""Shared LLM factory for PaperMind.

All agents and orchestrator nodes use make_llm() to get a ChatOpenAI instance.
reasoning_split=True tells MiniMax to separate <think> content from the answer,
so content only contains the final response.
"""

from langchain_openai import ChatOpenAI

from shared.config import Config


def make_llm(
    config: Config,
    temperature: float = 0.3,
    max_tokens: int | None = None,
) -> ChatOpenAI:
    """Return a ChatOpenAI instance configured for MiniMax (or any OpenAI-compatible API)."""
    return ChatOpenAI(
        api_key=config.llm_api_key,
        base_url=config.llm_base_url,
        model=config.llm_model,
        temperature=temperature,
        max_tokens=max_tokens or config.llm_max_tokens,
        max_retries=3,
        extra_body={"reasoning_split": True},
    )
