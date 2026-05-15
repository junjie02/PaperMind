#!/usr/bin/env python3
"""
DeepResearch -- 自动化文献调研 Agent

用法:
    deepresearch "diffusion models for video generation"
    deepresearch "attention mechanisms in vision transformers" -n 50 -w 4
    deepresearch "vision transformers" --out runs/my-survey --resume
"""

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from deepresearch.config import Config
from deepresearch.orchestrator import Orchestrator
from deepresearch.text_utils import slugify

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("deepresearch")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RUNS_ROOT = PROJECT_ROOT / "runs"


def _resolve_run_dir(out: str | None, question: str, resume: bool) -> Path:
    if out:
        run_dir = Path(out).expanduser().resolve()
        if resume and not run_dir.exists():
            raise SystemExit(f"--resume 指定的目录不存在: {run_dir}")
    else:
        if resume:
            raise SystemExit("--resume 必须配合 --out DIR 使用")
        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        run_dir = RUNS_ROOT / f"{timestamp}-{slugify(question)}"
    (run_dir / "pdfs").mkdir(parents=True, exist_ok=True)
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="DeepResearch - 自动化文献调研 Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  deepresearch "diffusion models for video generation"
  deepresearch "attention mechanisms in vision transformers" -n 50 -w 4
  deepresearch "vision transformers" --out runs/my-survey --resume
        """,
    )
    parser.add_argument("question", help="研究方向或问题")
    parser.add_argument("-n", "--num-papers", type=int, default=None,
                        help="目标论文数量 (默认: 30)")
    parser.add_argument("-w", "--workers", type=int, default=None,
                        help="并行 Worker 数 (默认: 1)")
    parser.add_argument("--out", default=None,
                        help="运行输出目录；不传则自动生成 runs/{时间戳}-{slug}/")
    parser.add_argument("--json", action="store_true",
                        help="以 JSON 输出论文元信息")
    parser.add_argument("--resume", action="store_true",
                        help="从已有运行目录恢复（必须配合 --out 使用）")
    parser.add_argument("--skip-pdf", action="store_true",
                        help="跳过 PDF 下载阶段")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="开启 DEBUG 日志")
    return parser


async def run_survey(args: argparse.Namespace) -> int:
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    config = Config()

    run_dir = _resolve_run_dir(args.out, args.question, args.resume)
    config.db_path = str(run_dir / "papers.db")
    config.pdf_dir = str(run_dir / "pdfs")
    logger.info("运行目录: %s", run_dir)

    orchestrator = Orchestrator(config)
    try:
        papers = await orchestrator.run(
            research_question=args.question,
            target_count=args.num_papers,
            workers_per_round=args.workers,
            resume=args.resume,
            skip_pdf=args.skip_pdf,
        )

        if args.json:
            payload = [
                {
                    "paper_id": p.paper_id,
                    "title": p.title,
                    "authors": p.authors,
                    "abstract": p.abstract,
                    "overview": p.overview,
                    "source": p.source,
                    "source_url": p.source_url,
                    "venue": p.venue,
                    "arxiv_id": p.arxiv_id,
                    "search_direction": p.search_direction,
                    "published_at": p.published_at,
                    "categories": p.categories,
                    "primary_class": p.primary_class,
                    "bibtex": p.bibtex,
                    "abs_url": p.abs_url,
                    "pdf_url": p.pdf_url,
                    "artifact_rel_path": p.artifact_rel_path,
                    "relevance_score": p.relevance_score,
                }
                for p in papers
            ]
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return 0

        pdf_count = sum(1 for p in papers if p.arxiv_id and p.artifact_rel_path)
        md_count = sum(1 for p in papers if not p.arxiv_id and p.artifact_rel_path)
        arxiv_total = sum(1 for p in papers if p.arxiv_id)
        non_arxiv_total = len(papers) - arxiv_total
        print(f"\n{'='*60}")
        print(f"调研完成: 共收集 {len(papers)} 篇论文")
        print(f"运行目录: {run_dir}")
        print(f"数据库:   {config.db_path}")
        if args.skip_pdf:
            print(f"产物目录: {config.pdf_dir} (已跳过)")
        else:
            print(
                f"产物目录: {config.pdf_dir} "
                f"(PDF {pdf_count}/{arxiv_total}, MD {md_count}/{non_arxiv_total})"
            )
        print(f"{'='*60}")

        if papers:
            groups: dict[str, list] = {}
            for p in papers:
                groups.setdefault(p.search_direction or "(未分类)", []).append(p)

            print("\n按方向分组:")
            for direction, items in groups.items():
                items_sorted = sorted(
                    items,
                    key=lambda p: (p.relevance_score, p.published_at or ""),
                    reverse=True,
                )
                print(f"\n  ▶ {direction}  ({len(items_sorted)} 篇)")
                for i, p in enumerate(items_sorted[:8]):
                    tag = f"[arxiv:{p.arxiv_id}]" if p.arxiv_id else f"[{p.source}]"
                    print(f"    {i+1}. {p.title}  {tag}")
                    if p.source_url:
                        print(f"       url: {p.source_url}")
                    if p.artifact_rel_path:
                        print(f"       file: {p.artifact_rel_path}")
                if len(items_sorted) > 8:
                    print(f"    ... 还有 {len(items_sorted) - 8} 篇")
    finally:
        await orchestrator.close()

    return 0


def main():
    parser = build_parser()
    args = parser.parse_args()
    sys.exit(asyncio.run(run_survey(args)))


if __name__ == "__main__":
    main()
