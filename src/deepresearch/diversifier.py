import logging

from deepresearch.llm_client import LLMClient
from deepresearch.models import SearchDirection

logger = logging.getLogger(__name__)


class SearchDiversifier:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    async def generate(
        self, question: str, n: int, exclude_ids: set[str]
    ) -> list[SearchDirection]:
        exclude_str = ", ".join(sorted(exclude_ids)[:50]) or "（尚无）"

        prompt = f"""你是一位研究馆员，擅长设计文献检索策略。

研究课题："{question}"

请为这个课题设计 {n} 个互补的文献搜索方向。每个方向应覆盖该领域不同且不重叠的子领域，以最大化文献覆盖面。

已收集论文的 arXiv ID（不要再找这些）：{exclude_str}

每个方向需包含：
1. 一段清晰具体的中文描述
2. 2-4 个英文搜索关键词（含 "arxiv" 关键字，便于搜索学术论文）

严格按以下 JSON 格式返回：
{{"directions": [{{"direction": "这个方向的描述", "search_queries": ["search query 1 arxiv", "search query 2 arxiv"]}}, ...]}}

示例（研究课题 "vision transformer 中的注意力机制"）：
{{
  "directions": [
    {{"direction": "原始 ViT 及早期视觉 Transformer 论文", "search_queries": ["vision transformer original paper arxiv", "ViT image classification arxiv"]}},
    {{"direction": "窗口注意力和局部注意力机制", "search_queries": ["Swin transformer arxiv", "window attention vision transformer arxiv", "local attention ViT arxiv"]}},
    {{"direction": "线性复杂度和高效注意力", "search_queries": ["linear attention vision transformer arxiv", "efficient self-attention arxiv", "Performer Linformer vision arxiv"]}},
    {{"direction": "通道注意力和混合注意力机制", "search_queries": ["channel attention vision arxiv", "spatial channel hybrid attention arxiv", "CBAM attention arxiv"]}}
  ]
}}"""

        messages = [
            {
                "role": "system",
                "content": "你是一位研究馆员。你的任务是设计多样化、不重叠的文献检索策略。只返回有效的 JSON。",
            },
            {"role": "user", "content": prompt},
        ]

        try:
            data = await self.llm.chat_json(messages, max_tokens=2000)
            directions = [SearchDirection(**d) for d in data.get("directions", [])]
            logger.info(f"生成了 {len(directions)} 个搜索方向")
            for i, d in enumerate(directions):
                logger.info(f"  方向{i}: {d.direction}")
                logger.info(f"    搜索词: {d.search_queries}")
            if len(directions) < n:
                logger.warning(f"只得到 {len(directions)} 个方向，期望 {n}")
            return directions[:n]
        except Exception as e:
            logger.error(f"生成搜索方向失败: {e}")
            return [
                SearchDirection(
                    direction=question,
                    search_queries=[f"{question} arxiv", f"{question} survey arxiv"],
                )
            ]
