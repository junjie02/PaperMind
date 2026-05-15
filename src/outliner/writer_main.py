#!/usr/bin/env python3
"""
index  — 提取论文文本并向量化存入 FAISS
write  — 调用 DeepSeek 并发写作 + RAG 验证

用法:
    write index --in runs/20260515T155426-agent-安全研究
    write write --in runs/20260515T155426-agent-安全研究 --skip-index
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from deepresearch.config import Config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("writer")


def _resolve_run_dir(in_dir: str) -> Path:
    run_dir = Path(in_dir).expanduser().resolve()
    if not run_dir.exists():
        logger.error("运行目录不存在: %s", run_dir)
        sys.exit(1)
    if not (run_dir / "papers.db").exists():
        logger.error("papers.db 不存在: %s", run_dir / "papers.db")
        sys.exit(1)
    return run_dir


# ---------------------------------------------------------------------------
# index subcommand
# ---------------------------------------------------------------------------

def cmd_index(args: argparse.Namespace):
    from .indexer import build_index_from_run

    config = Config()
    run_dir = _resolve_run_dir(args.in_dir)
    logger.info("运行目录: %s", run_dir)

    logger.info("=" * 50)
    logger.info("PDF 文本提取 + 切分 + 向量化")
    logger.info("=" * 50)
    try:
        index_path = build_index_from_run(run_dir, embedding_model=config.embedding_model)
    except ValueError as e:
        logger.error("%s", e)
        sys.exit(1)
    logger.info("完成。FAISS index: %s", index_path)


# ---------------------------------------------------------------------------
# write subcommand
# ---------------------------------------------------------------------------

async def _do_write(args: argparse.Namespace):
    from .db import count_papers
    from .indexer import build_index_from_run
    from .review_orchestrator import ReviewOrchestrator

    config = Config()
    if args.model:
        config.writer_model = args.model

    run_dir = _resolve_run_dir(args.in_dir)
    outline_path = run_dir / "outline.md"

    if not outline_path.exists():
        logger.error("outline.md 不存在: %s", outline_path)
        sys.exit(1)

    outline = outline_path.read_text(encoding="utf-8")
    n_papers = count_papers(run_dir / "papers.db")
    logger.info("运行目录: %s  论文数量: %d", run_dir, n_papers)

    if not args.skip_index:
        logger.info("=" * 50)
        logger.info("PDF 文本提取 + 切分 + 向量化")
        logger.info("=" * 50)
        try:
            build_index_from_run(run_dir, embedding_model=config.embedding_model)
        except ValueError as e:
            logger.error("%s", e)
            sys.exit(1)
    else:
        logger.info("跳过向量化 (--skip-index)")

    logger.info("=" * 50)
    logger.info("并发写作 (DeepSeek + FAISS RAG)")
    logger.info("=" * 50)

    orchestrator = ReviewOrchestrator(config, run_dir)
    review_md = await orchestrator.write_review(outline, n_papers)

    if not review_md:
        logger.error("写作未返回结果")
        sys.exit(1)

    out_path = Path(args.out) if args.out else run_dir / "review.md"
    out_path.write_text(review_md, encoding="utf-8")
    logger.info("综述已写出: %s (%d 字符)", out_path, len(review_md))


def cmd_write(args: argparse.Namespace):
    asyncio.run(_do_write(args))


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="write",
        description="PaperMind 写作工具：向量化 (index) 或撰写综述 (write)",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_index = sub.add_parser("index", help="提取 PDF 文本并建 FAISS 向量索引")
    p_index.add_argument("--in", dest="in_dir", required=True,
                         help="运行目录（含 papers.db 和 pdfs/）")

    p_write = sub.add_parser("write", help="调用 DeepSeek 并发写作 + RAG 验证")
    p_write.add_argument("--in", dest="in_dir", required=True,
                         help="运行目录（含 outline.md 和 papers.db）")
    p_write.add_argument("--skip-index", action="store_true",
                         help="跳过向量化，复用已有 FAISS index")
    p_write.add_argument("--model", default=None,
                         help="写作使用的 DeepSeek 模型（覆盖 WRITER_MODEL）")
    p_write.add_argument("--out", default=None,
                         help="输出文件路径 (默认: {in_dir}/review.md)")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.cmd == "index":
        cmd_index(args)
    elif args.cmd == "write":
        cmd_write(args)


if __name__ == "__main__":
    main()
