"""SectionWriter: write-verify loop for a single outline section.

Each instance handles one section independently:
  1. Dual-path RAG retrieval
  2. DeepSeek write
  3. Single-path RAG retrieval on draft
  4. DeepSeek verify → PASS or FAIL+feedback
  5. Loop until PASS or max_retries exhausted
"""

import asyncio
import logging

from openai import AsyncOpenAI

from deepresearch.config import Config

from .retriever import Retriever, Section, format_chunks_for_llm
from .writer_prompts import (
    FEEDBACK_BLOCK_TEMPLATE,
    VERIFY_SYSTEM_PROMPT,
    VERIFY_USER_PROMPT,
    WRITE_SYSTEM_PROMPT,
    WRITE_USER_PROMPT,
)

logger = logging.getLogger(__name__)


class SectionWriter:
    def __init__(
        self,
        config: Config,
        retriever: Retriever,
        semaphore: asyncio.Semaphore,
        client: AsyncOpenAI,
    ):
        self.retriever = retriever
        self.semaphore = semaphore
        self.max_retries = config.writer_max_retries
        self._model = config.writer_model or config.llm_model
        self._max_tokens = config.llm_max_tokens
        self._client = client

    async def write_section(self, section: Section) -> str:
        """Run the write-verify loop for one section. Returns final draft."""
        feedback = ""
        draft = ""

        for attempt in range(self.max_retries):
            logger.info("[%s] attempt %d/%d", section.title, attempt + 1, self.max_retries)

            chunks = await self.retriever.dual_search(section, top_k=6)
            logger.info("[%s] retrieved %d chunks", section.title, len(chunks))

            draft = await self._write(section, chunks, feedback)
            if not draft:
                logger.warning("[%s] write returned empty, skipping verify", section.title)
                continue

            verify_chunks = self.retriever.search(draft[:600], top_k=6)
            verdict = await self._verify(draft, verify_chunks)
            logger.info("[%s] verdict: %s", section.title, verdict[:60])

            if verdict.strip().upper().startswith("PASS"):
                logger.info("[%s] PASS on attempt %d", section.title, attempt + 1)
                return draft

            feedback = verdict[4:].strip() if verdict.upper().startswith("FAIL") else verdict

        logger.warning("[%s] max retries reached, returning last draft", section.title)
        return draft

    async def _write(self, section: Section, chunks, feedback: str) -> str:
        feedback_block = FEEDBACK_BLOCK_TEMPLATE.format(feedback=feedback) if feedback else ""
        chunks_text = format_chunks_for_llm(chunks) if chunks else "（未检索到相关片段）"
        user_msg = WRITE_USER_PROMPT.format(
            section_outline=section.outline_text,
            chunks_text=chunks_text,
            feedback_block=feedback_block,
        )
        return await self._call_llm(WRITE_SYSTEM_PROMPT, user_msg, temperature=0.4)

    async def _verify(self, draft: str, verify_chunks) -> str:
        verify_chunks_text = (
            format_chunks_for_llm(verify_chunks) if verify_chunks else "（未检索到核查片段）"
        )
        user_msg = VERIFY_USER_PROMPT.format(
            draft=draft,
            verify_chunks_text=verify_chunks_text,
        )
        return await self._call_llm(VERIFY_SYSTEM_PROMPT, user_msg, temperature=0)

    async def _call_llm(self, system: str, user: str, temperature: float) -> str:
        async with self.semaphore:
            try:
                resp = await self._client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=temperature,
                    max_tokens=self._max_tokens,
                )
                return resp.choices[0].message.content or ""
            except Exception as e:
                logger.error("LLM call failed: %s", e)
                return ""
