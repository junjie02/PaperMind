#!/usr/bin/env python3
"""
PM (PaperMind) — 主 Agent + Multi Sub-Agent 文献综述生成

用法:
    PM "agent 安全研究" -n 15
    PM "diffusion models for video" -n 30
    PM "vision transformers" -n 10
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

# Suppress noisy third-party loggers
logging.getLogger("ddgs.ddgs").setLevel(logging.WARNING)
logging.getLogger("primp").setLevel(logging.WARNING)
logging.getLogger("langchain_openai.chat_models._client_utils").setLevel(logging.WARNING)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="PM",
        description="PaperMind — 主 Agent + Multi Sub-Agent 文献综述生成",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  PM "agent 安全研究" -n 15
  PM "diffusion models for video" -n 30
  PM "vision transformers" -n 10
        """,
    )
    parser.add_argument("question", nargs="?", default=None, help="研究课题")
    parser.add_argument("-n", "--num-papers", type=int, default=50,
                        help="目标参考文献总数（默认 20）")
    parser.add_argument("--out", default=None,
                        help="运行输出目录；不传则自动生成 runs/{时间戳}-{slug}/")
    parser.add_argument("--resume", default=None,
                        help="从已有运行目录恢复，指定起始阶段（如 build_index、write_sections）")
    parser.add_argument("--skip-review", action="store_true",
                        help="跳过 reviewer 评审阶段，writer 写完直接进入 polish")
    parser.add_argument("--pdf", action="store_true",
                        help="运行完成后自动将 survey.md 转为 PDF")
    parser.add_argument("--to-pdf", default=None, metavar="MD_FILE",
                        help="单独将指定 md 文件转为 PDF（不运行 pipeline）")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="开启 DEBUG 日志")
    return parser


def _slugify(text: str) -> str:
    import re
    text = text.lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "-", text)
    return text[:40].strip("-")


def _resolve_run_dir(out: str | None, question: str) -> Path:
    if out:
        run_dir = Path(out).expanduser().resolve()
    else:
        project_root = Path(__file__).resolve().parent.parent.parent
        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        run_dir = project_root / "runs" / f"{timestamp}-{_slugify(question)}"
    (run_dir / "pdfs").mkdir(parents=True, exist_ok=True)
    return run_dir


def _load_resume_state(run_dir: Path) -> dict:
    """Load existing state from a run directory for resuming."""
    import json
    state: dict = {}

    # Restore outline from data/main_agent.json (parse from LLM output)
    outline_md = run_dir / "outline.md"
    if outline_md.exists():
        # Parse outline.md back into structured format
        chapters = []
        current_chapter: dict | None = None
        for line in outline_md.read_text(encoding="utf-8").splitlines():
            if line.startswith("## "):
                if current_chapter:
                    chapters.append(current_chapter)
                current_chapter = {"title": line[3:].strip(), "description": "", "sub_questions": []}
            elif line.startswith("- ") and current_chapter is not None:
                # Strip trailing （目标 N 篇）
                import re
                sq = re.sub(r"\s*（目标\s*\d+\s*篇）\s*$", "", line[2:].strip())
                current_chapter["sub_questions"].append(sq)
            elif line and current_chapter is not None and not current_chapter["description"]:
                current_chapter["description"] = line.strip()
        if current_chapter:
            chapters.append(current_chapter)
        if chapters:
            state["research_outline"] = chapters
            logger.info("恢复大纲: %d 章节", len(chapters))

    # Restore main_agent conversation history
    main_agent_log = run_dir / "data" / "main_agent.json"
    if main_agent_log.exists():
        try:
            records = json.loads(main_agent_log.read_text(encoding="utf-8"))
            if records:
                last = records[-1]
                history = last.get("history", [])
                # Add the last exchange
                if last.get("input") and last.get("output"):
                    history = history + [
                        {"role": "human", "content": last["input"]},
                        {"role": "ai", "content": last["output"]},
                    ]
                state["agent_messages"] = history
        except Exception as e:
            logger.warning("恢复主 Agent 历史失败: %s", e)

    return state


async def _run_pipeline(question: str, run_dir: Path, num_papers: int = 20, resume_from: str | None = None, skip_review: bool = False) -> None:
    from orchestrator.graph import build_graph
    from shared.config import Config

    config = Config()
    if not config.llm_api_key:
        logger.error("未设置 OPENAI_API_KEY，无法运行")
        sys.exit(1)

    graph = build_graph(config)
    initial_state = {
        "research_topic": question,
        "run_dir": str(run_dir),
        "db_path": str(run_dir / "papers.db"),
        "target_papers": num_papers,
        "skip_review": skip_review,
        "revision_count": 0,
        "max_revisions": 2,
    }

    if resume_from:
        from orchestrator.graph import build_graph_from
        graph = build_graph_from(config, resume_from)
        # Restore state from run directory
        initial_state.update(_load_resume_state(run_dir))
        logger.info("从阶段 [%s] 恢复运行", resume_from)

    config_lg = {"configurable": {"thread_id": f"pm-{run_dir.name}"}}
    final = await graph.ainvoke(initial_state, config_lg)

    output_path = final.get("output_path", "")
    if output_path:
        logger.info("综述已生成: %s", output_path)
    else:
        logger.warning("综述生成可能未完成，请检查运行目录: %s", run_dir)


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Handle --to-pdf: standalone conversion, no pipeline
    if args.to_pdf:
        from papermind.pdf_export import md_to_pdf
        pdf_path = md_to_pdf(args.to_pdf)
        print(f"PDF 已生成: {pdf_path}")
        return

    if not args.question:
        parser.error("请提供研究课题（或使用 --to-pdf 转换已有文件）")

    run_dir = _resolve_run_dir(args.out, args.question)
    logger.info("PaperMind 启动")
    logger.info("研究课题: %s", args.question)
    logger.info("目标文献: %d 篇", args.num_papers)
    logger.info("运行目录: %s", run_dir)

    asyncio.run(_run_pipeline(args.question, run_dir, num_papers=args.num_papers, resume_from=args.resume, skip_review=args.skip_review))

    # Auto PDF export if --pdf flag is set
    if args.pdf:
        survey_path = run_dir / "survey.md"
        if survey_path.exists():
            from papermind.pdf_export import md_to_pdf
            pdf_path = md_to_pdf(survey_path)
            print(f"PDF 已生成: {pdf_path}")

    print(f"\n{'=' * 60}")
    print("PaperMind 完成")
    print(f"运行目录: {run_dir}")
    db_path = run_dir / "papers.db"
    if db_path.exists():
        print(f"数据库:   {db_path}")
    survey_path = run_dir / "survey.md"
    if survey_path.exists():
        print(f"综述:     {survey_path}")
    pdf_path = run_dir / "survey.pdf"
    if pdf_path.exists():
        print(f"PDF:      {pdf_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
