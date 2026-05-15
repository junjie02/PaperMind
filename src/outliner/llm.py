from langchain_openai import ChatOpenAI

from deepresearch.config import Config


def build_chat_model(
    config: Config,
    *,
    temperature: float,
    json_mode: bool = False,
) -> ChatOpenAI:
    """Construct a ChatOpenAI client from deepresearch.Config.

    The Config field names use the DEEPSEEK_ namespace but the underlying
    interface is OpenAI-compatible (DeepSeek, OpenAI, local Ollama all work).
    """
    kwargs = dict(
        model=config.llm_model,
        api_key=config.llm_api_key,
        base_url=config.llm_base_url,
        temperature=temperature,
        max_tokens=config.llm_max_tokens,
    )
    if json_mode:
        kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}
    return ChatOpenAI(**kwargs)
