#!/usr/bin/env python3
"""
DeepResearch -- 自动化文献调研 Agent

用法:
    deepresearch "diffusion models for video generation"
    deepresearch "attention mechanisms in vision transformers" -n 50 -w 4
    deepresearch "large language model reasoning" --db llm_reasoning.db --resume
"""

import argparse
import asyncio
import json
import logging
import sys

from deepresearch.config import Config
from deepresearch.orchestrator import Orchestrator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("deepresearch")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="DeepResearch - 自动化文献调研 Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  deepresearch "diffusion models for video generation"
  deepresearch "attention mechanisms in vision transformers" -n 50 -w 4
  deepresearch "large language model reasoning" --db llm_reasoning.db --resume
        """,
    )
    parser.add_argument("question", help="研究方向或问题")
    parser.add_argument("-n", "--num-papers", type=int, default=None,
                        help="目标论文数量 (默认: 30)")
    parser.add_argument("-w", "--workers", type=int, default=None,
                        help="每轮并行 Worker 数 (默认: 4)")
    parser.add_argument("--db", default="papers.db",
                        help="SQLite 数据库路径 (默认: papers.db)")
    parser.add_argument("--backend", choices=["claude", "arxiv"], default=None,
                        help="搜索后端: claude 默认, arxiv 为直连 API fallback")
    parser.add_argument("--json", action="store_true",
                        help="以 JSON 输出论文元信息")
    parser.add_argument("--resume", action="store_true",
                        help="从已有数据库恢复，跳过已收集论文")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="开启 DEBUG 日志")
    return parser


async def run_survey(args: argparse.Namespace) -> int:
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    config = Config()
    if args.db:
        config.db_path = args.db
    if args.backend:
        config.search_backend = args.backend

    if config.search_backend != "arxiv" and not config.llm_api_key:
        logger.error("未设置 DEEPSEEK_API_KEY 环境变量")
        logger.error("Claude 后端需要此密钥；arxiv fallback 不需要 LLM。")
        logger.error("请在 .env 文件中配置: cp .env.example .env 然后编辑")
        return 1

    orchestrator = Orchestrator(config)
    try:
        papers = await orchestrator.run(
            research_question=args.question,
            target_count=args.num_papers,
            workers_per_round=args.workers,
            resume=args.resume,
        )

        if args.json:
            payload = [
                {
                    "arxiv_id": p.arxiv_id,
                    "title": p.title,
                    "authors": p.authors,
                    "abstract": p.abstract,
                    "overview": p.overview,
                    "published_at": p.published_at,
                    "categories": p.categories,
                    "primary_class": p.primary_class,
                    "bibtex": p.bibtex,
                    "abs_url": p.abs_url,
                    "pdf_url": p.pdf_url,
                    "relevance_score": p.relevance_score,
                }
                for p in papers
            ]
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return 0

        print(f"\n{'='*60}")
        print(f"调研完成: 共收集 {len(papers)} 篇论文")
        print(f"数据库: {config.db_path}")
        print(f"后端: {config.search_backend}")
        print(f"{'='*60}")

        if papers:
            print("\n论文元信息:")
            sorted_papers = sorted(
                papers,
                key=lambda p: (p.relevance_score, p.published_at or ""),
                reverse=True,
            )
            for i, p in enumerate(sorted_papers[:10]):
                print(f"  {i+1}. {p.title}")
                print(f"     arxiv_id: {p.arxiv_id}")
                print(f"     abs_url:  {p.abs_url}")
                print(f"     pdf_url:  {p.pdf_url}")
                if p.published_at:
                    print(f"     date:     {p.published_at}")
            if len(papers) > 10:
                print(f"  ... 还有 {len(papers) - 10} 篇")
    finally:
        await orchestrator.close()

    return 0


def main():
    parser = build_parser()
    args = parser.parse_args()
    sys.exit(asyncio.run(run_survey(args)))


if __name__ == "__main__":
    main()
