#!/usr/bin/env python3
"""
PM (PaperMind) — 一键完成文献搜集 + 大纲生成

用法:
    PM "agent 安全研究" -n 15
    PM "diffusion models for video" -n 30 -w 2 --skip-outline
    PM "vision transformers" -n 10 --skip-pdf
"""

import argparse
import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("papermind")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="PM",
        description="PaperMind — 一键完成文献搜集 + 大纲生成",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  PM "agent 安全研究" -n 15
  PM "diffusion models for video" -n 30 -w 2
  PM "vision transformers" -n 10 --skip-pdf
  PM "LLM reasoning" -n 20 --skip-outline
        """,
    )
    parser.add_argument("question", help="研究课题")
    parser.add_argument("-n", "--num-papers", type=int, default=30,
                        help="目标论文数量 (默认: 30)")
    parser.add_argument("-w", "--workers", type=int, default=1,
                        help="并行 Claude Code Worker 数 (默认: 1)")
    parser.add_argument("--out", default=None,
                        help="运行输出目录；不传则自动生成 runs/{时间戳}-{slug}/")
    parser.add_argument("--skip-pdf", action="store_true",
                        help="跳过 PDF/MD 产物下载")
    parser.add_argument("--skip-outline", action="store_true",
                        help="只搜集文献，不生成大纲")
    parser.add_argument("-s", "--sections", type=int, default=8,
                        help="大纲目标章节数 (默认: 8)")
    parser.add_argument("--subsections", type=int, default=2,
                        help="每章节子节数 (默认: 2)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="开启 DEBUG 日志")
    parser.add_argument("--skip-write", action="store_true",
                        help="跳过综述正文写作（Phase 3）")
    return parser


async def _run_deepresearch(question: str, num_papers: int, workers: int,
                            run_dir: Path, skip_pdf: bool) -> bool:
    """Phase 1: literature collection via Claude Code workers."""
    from deepresearch.config import Config
    from deepresearch.orchestrator import Orchestrator

    config = Config()
    config.db_path = str(run_dir / "papers.db")
    config.pdf_dir = str(run_dir / "pdfs")

    orchestrator = Orchestrator(config)
    try:
        papers = await orchestrator.run(
            research_question=question,
            target_count=num_papers,
            workers_per_round=workers,
            skip_pdf=skip_pdf,
        )
        logger.info("文献搜集完成: %d 篇", len(papers))
        return len(papers) > 0
    finally:
        await orchestrator.close()


def _run_outline(run_dir: Path, topic: str, sections: int, subsections: int) -> bool:
    """Phase 2: outline generation via LangGraph pipeline."""
    from deepresearch.config import Config

    from outliner.graph import build_graph

    config = Config()
    if not config.llm_api_key:
        logger.warning("未设置 DEEPSEEK_API_KEY，跳过大纲生成")
        return False

    class _Args:
        out = None
        recluster = True
        force = True

    initial_state = {
        "in_dir": str(run_dir),
        "topic": topic,
        "n_sections": sections,
        "n_subsections": subsections,
        "messages": [],
        "papers": [],
        "cluster_assignments": [],
        "cluster_skipped_reason": "",
        "paper_groups": [],
        "chapter_plan": [],
        "survey_title": "",
        "group_outlines": [],
        "final_outline": "",
        "references_md": "",
        "output_path": "",
        "revision_count": 0,
        "review_feedback": "",
    }

    _Args.out = str(run_dir / "outline.md")
    graph = build_graph(config, _Args)
    config_lg = {"configurable": {"thread_id": f"pm-{run_dir.name}"}}

    try:
        final = graph.invoke(initial_state, config_lg)
        logger.info("大纲已生成: %s", final.get("output_path", "?"))
        return True
    except Exception as e:
        logger.error("大纲生成失败: %s", e)
        return False


def _run_write(run_dir: Path) -> bool:
    """Phase 3: write full review via DeepSeek concurrent writers + RAG."""
    from deepresearch.config import Config

    from outliner.db import count_papers
    from outliner.indexer import build_index_from_run
    from outliner.review_orchestrator import ReviewOrchestrator

    config = Config()
    outline = (run_dir / "outline.md").read_text(encoding="utf-8")

    try:
        build_index_from_run(run_dir, embedding_model=config.embedding_model)
    except ValueError as e:
        logger.warning("跳过综述写作: %s", e)
        return False

    n_papers = count_papers(run_dir / "papers.db")
    orchestrator = ReviewOrchestrator(config, run_dir)
    review_md = asyncio.run(orchestrator.write_review(outline, n_papers))

    if review_md:
        (run_dir / "review.md").write_text(review_md, encoding="utf-8")
        logger.info("综述已生成: %s", run_dir / "review.md")
        return True
    else:
        logger.error("综述写作失败")
        return False


def _resolve_run_dir(out: str | None, question: str) -> Path:
    from deepresearch.text_utils import slugify

    if out:
        run_dir = Path(out).expanduser().resolve()
    else:
        project_root = Path(__file__).resolve().parent.parent.parent
        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        run_dir = project_root / "runs" / f"{timestamp}-{slugify(question)}"
    (run_dir / "pdfs").mkdir(parents=True, exist_ok=True)
    return run_dir


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    run_dir = _resolve_run_dir(args.out, args.question)
    logger.info("PaperMind 启动")
    logger.info("研究课题: %s", args.question)
    logger.info("运行目录: %s", run_dir)

    # Phase 1: 文献搜集
    logger.info("=" * 50)
    logger.info("Phase 1: 文献搜集 (Claude Code)")
    logger.info("=" * 50)
    has_papers = asyncio.run(
        _run_deepresearch(args.question, args.num_papers, args.workers,
                          run_dir, args.skip_pdf)
    )

    if not has_papers:
        logger.error("未搜集到任何论文，流程终止")
        sys.exit(1)

    # Phase 2: 大纲生成
    if args.skip_outline:
        logger.info("跳过大纲生成 (--skip-outline)")
    else:
        logger.info("=" * 50)
        logger.info("Phase 2: 大纲生成 (LangGraph + DeepSeek)")
        logger.info("=" * 50)
        _run_outline(run_dir, args.question, args.sections, args.subsections)

    # Phase 3: 综述写作
    outline_path = run_dir / "outline.md"
    if args.skip_outline or args.skip_write:
        logger.info("跳过综述写作 (--skip-write)")
    elif not outline_path.exists():
        logger.warning("outline.md 不存在，跳过综述写作")
    else:
        logger.info("=" * 50)
        logger.info("Phase 3: 综述写作 (DeepSeek + FAISS RAG)")
        logger.info("=" * 50)
        _run_write(run_dir)

    # 最终汇总
    print(f"\n{'=' * 60}")
    print(f"PaperMind 完成")
    print(f"运行目录: {run_dir}")
    print(f"数据库:   {run_dir / 'papers.db'}")
    print(f"产物目录: {run_dir / 'pdfs'}")
    outline_path = run_dir / "outline.md"
    if outline_path.exists():
        print(f"大纲:     {outline_path}")
    review_path = run_dir / "review.md"
    if review_path.exists():
        print(f"综述:     {review_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
