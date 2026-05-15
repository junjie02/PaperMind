#!/usr/bin/env python3
"""
outline -- 从 deepresearch 运行目录生成文献综述大纲

用法:
    outline --in runs/20260515T134500-attention-in-vit
    outline --in runs/my-survey --out custom.md --force
    outline --in runs/my-survey --recluster -s 10
"""

import argparse
import logging
import re
import sys
from pathlib import Path

from deepresearch.config import Config

from .graph import build_graph

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("outliner")

_TS_SLUG_RE = re.compile(r"^\d{8}T\d{6}-(.+)$")


def _topic_from_dirname(dirname: str) -> str:
    """Derive a human-readable topic from the run-dir name.

    E.g. '20260515T134500-attention-in-vision-transformers'
    → 'attention in vision transformers'
    """
    m = _TS_SLUG_RE.match(dirname)
    slug = m.group(1) if m else dirname
    return slug.replace("-", " ")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="outline - 从 deepresearch 运行目录生成文献综述大纲",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  outline --in runs/20260515T134500-attention-in-vit
  outline --in runs/my-survey --out custom.md --force
  outline --in runs/my-survey --recluster -s 10
        """,
    )
    parser.add_argument("--in", dest="in_dir", required=True,
                        help="已完成的 deepresearch 运行目录（含 papers.db）")
    parser.add_argument("--out", default=None,
                        help="输出文件路径（默认: {in_dir}/outline.md）")
    parser.add_argument("--force", action="store_true",
                        help="覆盖已有 outline.md")
    parser.add_argument("--recluster", action="store_true",
                        help="强制重新聚类（即使 DB 已有分组）")
    parser.add_argument("-s", "--sections", type=int, default=8,
                        help="目标章节数（默认: 8）")
    parser.add_argument("--subsections", type=int, default=2,
                        help="每章节子节数（默认: 2）")
    parser.add_argument("--topic", default=None,
                        help="综述主题（默认从目录名推导）")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="开启 DEBUG 日志")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    config = Config()
    if not config.llm_api_key:
        sys.exit("错误: 未设置 DEEPSEEK_API_KEY（.env 或环境变量）")

    in_dir = Path(args.in_dir).resolve()
    if not (in_dir / "papers.db").exists():
        sys.exit(f"错误: 找不到 papers.db: {in_dir}")

    out_path = Path(args.out) if args.out else in_dir / "outline.md"
    if out_path.exists() and not args.force:
        sys.exit(f"{out_path} 已存在，加 --force 覆盖")

    topic = args.topic or _topic_from_dirname(in_dir.name)
    logger.info("综述主题: %s", topic)
    logger.info("输入目录: %s", in_dir)
    logger.info("输出文件: %s", out_path)

    initial_state = {
        "in_dir": str(in_dir),
        "topic": topic,
        "n_sections": args.sections,
        "n_subsections": args.subsections,
        "messages": [],
        "papers": [],
        "cluster_assignments": [],
        "cluster_skipped_reason": "",
        "paper_groups": [],
        "group_outlines": [],
        "final_outline": "",
        "references_md": "",
        "output_path": "",
        "revision_count": 0,
        "review_feedback": "",
    }

    graph = build_graph(config, args)
    config_lg = {"configurable": {"thread_id": f"outline-{in_dir.name}"}}

    final = graph.invoke(initial_state, config_lg)

    print(f"\n大纲已生成: {final['output_path']}")
    if final.get("cluster_skipped_reason"):
        print(f"(聚类已跳过: {final['cluster_skipped_reason']})")


if __name__ == "__main__":
    main()
