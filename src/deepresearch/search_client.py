import asyncio
import json
import logging
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

from deepresearch.models import PaperRecord

logger = logging.getLogger(__name__)

SCHEMA_PATH = Path(__file__).resolve().parent.parent.parent / "papers.schema.json"
LOG_DIR = Path(__file__).resolve().parent.parent.parent / "data"


class SearchClient:
    def __init__(self, max_timeout: int = 900):
        self.max_timeout = max_timeout
        self._call_counter = 0
        with open(SCHEMA_PATH, encoding="utf-8") as f:
            self._schema_str = json.dumps(json.load(f), ensure_ascii=False)

    async def exec(self, prompt: str, task_info: dict | None = None) -> dict:
        return await self._exec_claude(prompt, task_info)

    async def _exec_claude(self, prompt: str, task_info: dict | None) -> dict:
        self._call_counter += 1
        call_id = self._call_counter
        task_info = task_info or {}
        started_at = time.monotonic()

        claude_cmd = [
            "claude", "--print",
            "--verbose",
            "--output-format", "stream-json",
            "--json-schema", self._schema_str,
            "--no-session-persistence",
            "--dangerously-skip-permissions",
            prompt,
        ]

        cmd = ["unbuffer", *claude_cmd] if shutil.which("unbuffer") else claude_cmd

        logger.info(f"执行 Claude Code: prompt_len={len(prompt)}")

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError:
            logger.error("claude 命令不存在")
            result = {"papers": [], "summary": "claude not found", "search_queries_used": []}
            self._save_log(call_id, "claude", started_at, prompt, task_info, -1, result, "", "")
            return result

        stdout_lines = []
        stderr_lines = []

        async def _read_stdout():
            if proc.stdout:
                async for line in proc.stdout:
                    line_text = line.decode(errors="replace").strip()
                    if not line_text:
                        continue
                    stdout_lines.append(line_text)
                    self._log_event(line_text)

        async def _read_stderr():
            if proc.stderr:
                async for line in proc.stderr:
                    line_text = line.decode(errors="replace").strip()
                    if not line_text:
                        continue
                    stderr_lines.append(line_text)
                    logger.warning(f"  [Claude stderr] {line_text[:500]}")

        stdout_task = asyncio.create_task(_read_stdout())
        stderr_task = asyncio.create_task(_read_stderr())

        try:
            await asyncio.wait_for(proc.wait(), timeout=self.max_timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            logger.error("Claude Code 超时")
            await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)
            result = {"papers": [], "summary": "timeout", "search_queries_used": []}
            self._save_log(call_id, "claude", started_at, prompt, task_info, -1, result,
                           "\n".join(stdout_lines), "\n".join(stderr_lines))
            return result
        finally:
            await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)

        exit_code = proc.returncode
        stdout_text = "\n".join(stdout_lines)
        stderr_text = "\n".join(stderr_lines)

        if exit_code != 0:
            logger.error(f"Claude Code 返回非零: {exit_code}")
            # 检查是否有 error_max_budget_usd
            for line in stdout_lines:
                try:
                    ev = json.loads(line)
                    if ev.get("subtype") == "error_max_budget_usd":
                        logger.error(f"预算超限: cost={ev.get('total_cost_usd')}")
                except json.JSONDecodeError:
                    pass
            result = {"papers": [], "summary": f"exit {exit_code}", "search_queries_used": []}
            self._save_log(call_id, "claude", started_at, prompt, task_info, exit_code, result,
                           stdout_text, stderr_text)
            return result

        result = self._parse_stdout_json(stdout_text)
        self._save_log(call_id, "claude", started_at, prompt, task_info, exit_code, result,
                       stdout_text, stderr_text)
        return result

    def _log_event(self, line: str) -> None:
        """流式输出 Claude Code 的事件日志。"""
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            logger.info(f"  [Claude] {line[:500]}")
            return

        ev_type = ev.get("type", "")
        subtype = ev.get("subtype", "")

        if ev_type == "system":
            tools = ev.get("tools", [])
            tool_text = f", tools={len(tools)}" if tools else ""
            model = ev.get("model", "")
            logger.info(f"  [Claude system] {subtype or 'event'} model={model}{tool_text}")

        if ev_type == "assistant":
            msg = ev.get("message", {})
            content = msg.get("content", [])
            for block in content:
                block_type = block.get("type", "")
                if block_type == "text":
                    self._log_text("Claude", block.get("text", ""))
                elif block_type == "tool_use":
                    self._log_tool_use(block.get("name", "?"), block.get("input", {}))
                else:
                    logger.info(f"  [Claude assistant:{block_type}] {json.dumps(block, ensure_ascii=False)[:500]}")

        elif ev_type == "user":
            msg = ev.get("message", {})
            content = msg.get("content", [])
            for block in content:
                block_type = block.get("type", "")
                if block_type == "tool_result":
                    self._log_tool_result(block)
                elif block_type:
                    logger.info(f"  [Claude user:{block_type}] {json.dumps(block, ensure_ascii=False)[:500]}")

        elif ev_type == "tool_use":
            self._log_tool_use(ev.get("tool", ev.get("name", "?")), ev.get("input", {}))

        elif ev_type == "tool_result":
            self._log_tool_result(ev)

        elif ev_type == "result":
            if subtype == "success":
                logger.info(f"  [完成] {ev.get('duration_ms', 0)}ms, cost=${ev.get('total_cost_usd', 0):.4f}")
            elif subtype == "error_max_budget_usd":
                logger.error(f"  [预算超限] cost=${ev.get('total_cost_usd', 0):.4f}")
            else:
                logger.info(f"  [结束:{subtype}]")

        elif ev_type:
            logger.debug(f"  [Claude event:{ev_type}] {json.dumps(ev, ensure_ascii=False)[:500]}")

    def _log_text(self, label: str, text: str) -> None:
        for item in text.split("\n"):
            item = item.strip()
            if item:
                logger.info(f"  [{label}] {item[:500]}")

    def _log_tool_use(self, tool: str, inp: dict) -> None:
        if tool == "WebSearch":
            query = inp.get("query", inp.get("queries", ""))
            logger.info(f"  [搜索] {str(query)[:500]}")
        elif tool == "WebFetch":
            url = inp.get("url", "")
            logger.info(f"  [抓取] {url}")
        elif tool == "Bash":
            command = inp.get("command", "")
            logger.info(f"  [命令] {command[:500]}")
        elif tool in {"Read", "Glob", "Grep", "LS"}:
            logger.info(f"  [工具:{tool}] {json.dumps(inp, ensure_ascii=False)[:500]}")
        else:
            logger.info(f"  [工具:{tool}] {json.dumps(inp, ensure_ascii=False)[:500]}")

    def _log_tool_result(self, event: dict) -> None:
        content = event.get("content", event.get("output", ""))
        if isinstance(content, list):
            chunks = []
            for block in content:
                if isinstance(block, dict):
                    chunks.append(str(block.get("text", block)))
                else:
                    chunks.append(str(block))
            content = " ".join(chunks)
        preview = str(content).replace("\n", " ").strip()
        if preview:
            marker = "工具错误" if event.get("is_error") else "结果"
            logger.info(f"  [{marker}] {preview[:500]}")

    def _parse_stdout_json(self, text: str) -> dict:
        """从 Claude Code stdout 提取 JSON 结果。

        Claude --output-format json 每行一个 JSON 事件。
        result 字段中可能包含：裸 JSON、markdown 代码块、或混合文本。
        """
        result_candidates = []
        result_text = ""
        for line in reversed(text.strip().splitlines()):
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") == "result" and "result" in event:
                result_text = event["result"]
                result_candidates.append(result_text)
                break
            if event.get("type") == "assistant":
                for block in event.get("message", {}).get("content", []):
                    if block.get("type") == "text" and block.get("text"):
                        result_candidates.append(block["text"])

        if not result_candidates:
            logger.error(f"Claude 输出中未找到 type=result 事件: {text[:500]}")
            return {"papers": [], "summary": "parse error", "search_queries_used": []}

        errors = []
        for candidate in result_candidates:
            parsed, error = self._parse_json_candidate(candidate)
            if parsed is not None:
                return parsed
            errors.append(error)

        logger.error(f"Claude result 解析失败: {result_text[:500]} | errors={errors[:3]}")
        return {"papers": [], "summary": "parse error", "search_queries_used": []}

    def _parse_json_candidate(self, text: str) -> tuple[dict | None, str]:
        cleaned = self._strip_markdown_fence(text)
        decoder = json.JSONDecoder()

        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                return parsed, ""
        except json.JSONDecodeError:
            pass

        for start, char in enumerate(cleaned):
            if char != "{":
                continue
            try:
                parsed, _ = decoder.raw_decode(cleaned[start:])
            except json.JSONDecodeError as exc:
                last_error = f"pos={exc.pos}: {exc.msg}"
                continue
            if isinstance(parsed, dict):
                return parsed, ""

        return None, locals().get("last_error", "no JSON object found")

    def _strip_markdown_fence(self, text: str) -> str:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            first_newline = cleaned.find("\n")
            if first_newline >= 0:
                cleaned = cleaned[first_newline + 1:]
            else:
                cleaned = cleaned[3:]
        if cleaned.rstrip().endswith("```"):
            cleaned = cleaned.rstrip()[:-3]
        return cleaned.strip()

    def _save_log(self, call_id: int, backend: str, started_at: float,
                  prompt: str, task_info: dict, exit_code: int, result: dict,
                  stdout: str, stderr: str) -> None:
        try:
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            duration_ms = int((time.monotonic() - started_at) * 1000)
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")

            filename = LOG_DIR / f"{backend}-{ts}-{call_id:03d}.json"

            log_entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "backend": backend,
                "call_id": call_id,
                "round_num": task_info.get("round_num", 0),
                "worker_index": task_info.get("worker_index", 0),
                "search_direction": task_info.get("search_direction", ""),
                "duration_ms": duration_ms,
                "exit_code": exit_code,
                "input": {
                    "prompt": prompt,
                },
                "output": result,
                "diagnostics": {
                    "stdout_head": stdout[:2000] if stdout else "",
                    "stderr_head": stderr[:2000] if stderr else "",
                },
            }

            with open(filename, "w", encoding="utf-8") as f:
                json.dump(log_entry, f, ensure_ascii=False, indent=2)

            logger.info(f"日志已保存: {filename}")

        except Exception as e:
            logger.warning(f"保存日志失败: {e}")

    def parse_papers(self, result: dict, task) -> list[PaperRecord]:
        papers = []
        for item in result.get("papers", []):
            try:
                p = PaperRecord(
                    arxiv_id=item["arxiv_id"],
                    title=item["title"],
                    authors=item.get("authors", []),
                    abstract=item.get("abstract", ""),
                    overview=item.get("overview", ""),
                    published_at=item.get("published_at"),
                    categories=item.get("categories", []),
                    primary_class=item.get("primary_class"),
                    bibtex=item.get("bibtex", ""),
                    abs_url=item.get("abs_url", ""),
                    pdf_url=item.get("pdf_url", ""),
                    relevance_score=item.get("relevance_score", 3),
                    search_round=task.round_num,
                    worker_id=f"claude-{task.round_num}-{task.worker_index}",
                )
                papers.append(p)
            except Exception as e:
                logger.warning(f"解析论文条目失败: {e} | {item.get('arxiv_id', '?')}")
        return papers
