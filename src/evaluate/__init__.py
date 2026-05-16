"""PaperMind 评测脚本

对一次运行的输出进行质量评估，生成评测报告。

用法:
    python -m evaluate.metrics runs/20260516T145202-llm-推理加速
"""

from __future__ import annotations

import json
import re
import sqlite3
import sys
import time
from pathlib import Path
from collections import Counter


def evaluate_run(run_dir: Path) -> dict:
    """Evaluate a completed run and return metrics dict."""
    metrics: dict = {}

    survey_path = run_dir / "survey.md"
    db_path = run_dir / "papers.db"
    data_dir = run_dir / "data"

    if not survey_path.exists():
        print(f"ERROR: survey.md not found in {run_dir}")
        return metrics

    survey_text = survey_path.read_text(encoding="utf-8")

    # ── 文献质量指标 ──────────────────────────────────────────────────────
    metrics["文献质量"] = _eval_citation_quality(survey_text, db_path)

    # ── RAG 检索指标 ──────────────────────────────────────────────────────
    metrics["RAG检索"] = _eval_rag_metrics(data_dir, db_path, survey_text)

    # ── 内容质量指标 ──────────────────────────────────────────────────────
    metrics["内容质量"] = _eval_content_quality(survey_text, run_dir)

    # ── 生成效率指标 ──────────────────────────────────────────────────────
    metrics["生成效率"] = _eval_efficiency(data_dir, run_dir)

    return metrics


def _eval_citation_quality(survey_text: str, db_path: Path) -> dict:
    """文献覆盖率、幻觉引用率、引用密度"""
    result = {}

    # Count total papers in DB
    total_papers_in_db = 0
    if db_path.exists():
        conn = sqlite3.connect(str(db_path))
        total_papers_in_db = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
        conn.close()

    # Extract citations from survey (numbered [N] format)
    numbered_citations = set(re.findall(r"\[(\d+)\]", survey_text))
    # Extract [?] citations
    unmatched_citations = len(re.findall(r"\[\?\]", survey_text))
    # Also count [?, N] style
    unmatched_citations += len(re.findall(r"\[\?[,\s]", survey_text))

    total_citation_instances = len(re.findall(r"\[\d+(?:,\s*\d+)*\]", survey_text)) + unmatched_citations
    unique_papers_cited = len(numbered_citations)

    result["数据库论文总数"] = total_papers_in_db
    result["实际引用论文数"] = unique_papers_cited
    result["文献覆盖率"] = f"{unique_papers_cited}/{total_papers_in_db} = {unique_papers_cited/max(1,total_papers_in_db)*100:.1f}%"
    result["未匹配引用数"] = unmatched_citations
    result["幻觉引用率"] = f"{unmatched_citations}/{max(1,total_citation_instances)} = {unmatched_citations/max(1,total_citation_instances)*100:.1f}%"

    # Citation density: citations per 1000 chars
    text_length = len(survey_text)
    result["正文字数"] = text_length
    result["引用密度"] = f"{total_citation_instances / max(1, text_length) * 1000:.2f} 次/千字"

    return result


def _eval_rag_metrics(data_dir: Path, db_path: Path, survey_text: str) -> dict:
    """检索召回率、检索利用率"""
    result = {}

    if not data_dir.exists():
        result["说明"] = "data/ 目录不存在，无法评估 RAG 指标"
        return result

    # Collect all writer iteration logs
    writer_logs = list(data_dir.glob("writer-*_iterations.json"))
    total_rag_papers_returned = set()
    total_queries = 0

    for log_path in writer_logs:
        try:
            data = json.loads(log_path.read_text(encoding="utf-8"))
            iterations = data.get("iterations", [])
            for it in iterations:
                queries = it.get("queries", [])
                total_queries += len(queries)
                qrc = it.get("query_results_count", 0)
                total_rag_papers_returned.add(qrc)  # approximate
        except Exception:
            continue

    # Papers actually cited in survey
    cited_nums = set(re.findall(r"\[(\d+)\]", survey_text))
    papers_cited = len(cited_nums)

    # Papers in DB
    total_in_db = 0
    if db_path.exists():
        conn = sqlite3.connect(str(db_path))
        total_in_db = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
        conn.close()

    result["Writer总RAG查询次数"] = total_queries
    result["数据库论文总数"] = total_in_db
    result["实际引用论文数"] = papers_cited
    result["检索利用率"] = f"{papers_cited}/{total_in_db} = {papers_cited/max(1,total_in_db)*100:.1f}%" if total_in_db else "N/A"
    result["说明"] = "检索召回率需要人工标注相关论文集合才能精确计算，此处用引用数/总数近似"

    return result


def _eval_content_quality(survey_text: str, run_dir: Path) -> dict:
    """结构完整性、重复率"""
    result = {}

    lines = survey_text.split("\n")

    # Check structure
    h1_count = sum(1 for l in lines if l.startswith("# ") and not l.startswith("## "))
    h2_count = sum(1 for l in lines if l.startswith("## ") and not l.startswith("### "))
    h3_count = sum(1 for l in lines if l.startswith("### "))

    result["一级标题数"] = h1_count
    result["二级标题数"] = h2_count
    result["三级标题数"] = h3_count

    # Check for empty sections (## followed by ## without content)
    empty_sections = 0
    for i, line in enumerate(lines):
        if line.startswith("## "):
            # Look ahead for content before next heading
            has_content = False
            for j in range(i + 1, min(i + 5, len(lines))):
                if lines[j].strip() and not lines[j].startswith("#"):
                    has_content = True
                    break
                if lines[j].startswith("## "):
                    break
            if not has_content:
                empty_sections += 1

    result["空章节数"] = empty_sections
    result["结构完整性"] = "通过" if empty_sections == 0 and h2_count >= 3 else f"存在问题（空章节:{empty_sections}）"

    # Repetition detection: find repeated sentences (>30 chars)
    sentences = re.split(r"[。！？\n]", survey_text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 30]
    sentence_counts = Counter(sentences)
    repeated = {s: c for s, c in sentence_counts.items() if c > 1}

    result["总句数"] = len(sentences)
    result["重复句数"] = len(repeated)
    result["重复率"] = f"{len(repeated)/max(1,len(sentences))*100:.2f}%"
    if repeated:
        result["重复示例"] = list(repeated.keys())[:3]

    # Check for instruction leakage
    leakage_patterns = ["推荐统一使用", "避免在同一段落", "与第.*章标题保持一致", "请输出", "直接输出JSON"]
    leakage_count = 0
    for pattern in leakage_patterns:
        leakage_count += len(re.findall(pattern, survey_text))
    result["指令泄漏次数"] = leakage_count

    return result


def _eval_efficiency(data_dir: Path, run_dir: Path) -> dict:
    """LLM调用次数、总耗时"""
    result = {}

    if not data_dir.exists():
        result["说明"] = "data/ 目录不存在"
        return result

    # Count LLM calls from all agent logs
    total_llm_calls = 0

    # Main agent calls
    main_log = data_dir / "main_agent.json"
    if main_log.exists():
        try:
            records = json.loads(main_log.read_text(encoding="utf-8"))
            total_llm_calls += len(records)
            result["主Agent调用次数"] = len(records)
        except Exception:
            pass

    # Explorer iterations
    explorer_logs = list(data_dir.glob("explorer-*_iterations.json"))
    explorer_calls = 0
    for log_path in explorer_logs:
        try:
            data = json.loads(log_path.read_text(encoding="utf-8"))
            explorer_calls += len(data.get("iterations", []))
        except Exception:
            continue
    result["Explorer总迭代次数"] = explorer_calls
    total_llm_calls += explorer_calls

    # Researcher iterations
    researcher_logs = list(data_dir.glob("researcher-*_iterations.json"))
    researcher_calls = 0
    for log_path in researcher_logs:
        try:
            data = json.loads(log_path.read_text(encoding="utf-8"))
            researcher_calls += len(data.get("iterations", []))
        except Exception:
            continue
    result["Researcher总迭代次数"] = researcher_calls
    total_llm_calls += researcher_calls

    # Writer iterations
    writer_logs = list(data_dir.glob("writer-*_iterations.json"))
    writer_calls = 0
    for log_path in writer_logs:
        try:
            data = json.loads(log_path.read_text(encoding="utf-8"))
            writer_calls += len(data.get("iterations", []))
        except Exception:
            continue
    result["Writer总迭代次数"] = writer_calls
    total_llm_calls += writer_calls

    # Reviewer iterations
    reviewer_logs = list(data_dir.glob("reviewer-*_iterations.json"))
    reviewer_calls = 0
    for log_path in reviewer_logs:
        try:
            data = json.loads(log_path.read_text(encoding="utf-8"))
            reviewer_calls += len(data.get("iterations", []))
        except Exception:
            continue
    result["Reviewer总迭代次数"] = reviewer_calls
    total_llm_calls += reviewer_calls

    result["总LLM调用次数(估算)"] = total_llm_calls

    # Estimate total time from first and last log timestamps
    all_timestamps = []
    for log_path in data_dir.glob("*.json"):
        try:
            data = json.loads(log_path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                for record in data:
                    if "timestamp" in record:
                        all_timestamps.append(record["timestamp"])
            elif isinstance(data, dict):
                for it in data.get("iterations", []):
                    if "timestamp" in it:
                        all_timestamps.append(it["timestamp"])
        except Exception:
            continue

    if all_timestamps:
        all_timestamps.sort()
        result["首次记录时间"] = all_timestamps[0]
        result["最后记录时间"] = all_timestamps[-1]

    # Reviewer issues count
    reviewer_agent_logs = list(data_dir.glob("reviewer-*.json"))
    total_issues = 0
    total_reviewed = 0
    for log_path in reviewer_agent_logs:
        if "_iterations" in log_path.name:
            continue
        try:
            records = json.loads(log_path.read_text(encoding="utf-8"))
            for record in records:
                output = record.get("output", {})
                issues = output.get("issues", [])
                total_issues += len(issues)
                total_reviewed += 1
        except Exception:
            continue

    if total_reviewed:
        result["Reviewer审核章节数"] = total_reviewed
        result["Reviewer发现问题总数"] = total_issues
        result["需人工修改率"] = f"{total_issues}/{total_reviewed} = 平均每节{total_issues/total_reviewed:.1f}个问题"

    return result


def print_report(metrics: dict) -> None:
    """Pretty print the evaluation report."""
    print("\n" + "=" * 60)
    print("PaperMind 评测报告")
    print("=" * 60)

    for category, values in metrics.items():
        print(f"\n{'─' * 40}")
        print(f"  {category}")
        print(f"{'─' * 40}")
        if isinstance(values, dict):
            for k, v in values.items():
                if isinstance(v, list):
                    print(f"  {k}:")
                    for item in v[:3]:
                        print(f"    - {item[:80]}")
                else:
                    print(f"  {k}: {v}")
        else:
            print(f"  {values}")

    print("\n" + "=" * 60)


def main():
    if len(sys.argv) < 2:
        print("用法: python -m evaluate.metrics <run_dir>")
        print("示例: python -m evaluate.metrics runs/20260516T145202-llm-推理加速")
        sys.exit(1)

    run_dir = Path(sys.argv[1]).resolve()
    if not run_dir.exists():
        print(f"ERROR: 目录不存在: {run_dir}")
        sys.exit(1)

    metrics = evaluate_run(run_dir)
    print_report(metrics)

    # Save report as JSON
    report_path = run_dir / "evaluation_report.json"
    report_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n评测报告已保存: {report_path}")


if __name__ == "__main__":
    main()
