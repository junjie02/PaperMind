import json
import logging

from openai import AsyncOpenAI

from deepresearch.config import Config

logger = logging.getLogger(__name__)


class LLMClient:
    """OpenAI-compatible client (DeepSeek by default).

    Currently no module in this project consumes it — the Claude Code worker
    handles all reasoning. Kept as a hook for future LLM-driven pre/post
    processing (e.g. result re-ranking, summarisation, classification).
    """

    def __init__(self, config: Config):
        self.config = config
        self._client: AsyncOpenAI | None = None

    @property
    def client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=self.config.llm_api_key,
                base_url=self.config.llm_base_url,
            )
        return self._client

    @property
    def model(self) -> str:
        return self.config.llm_model

    @property
    def max_tokens(self) -> int:
        return self.config.llm_max_tokens

    async def chat(
        self,
        messages: list[dict],
        response_format: dict | None = None,
        max_tokens: int | None = None,
    ) -> str:
        kwargs = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens or self.max_tokens,
        }
        if response_format:
            kwargs["response_format"] = response_format

        response = await self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content

    async def chat_json(self, messages: list[dict], max_tokens: int | None = None) -> dict:
        fmt = {"type": "json_object"} if self.config.llm_json_mode else None
        text = await self.chat(
            messages=messages,
            response_format=fmt,
            max_tokens=max_tokens,
        )
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON from LLM response: {text[:500]}")
            raise
