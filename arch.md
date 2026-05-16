# PaperMind 架构文档

## 整体结构

```
用户输入 (PM "topic" -n 30)
    │
    ▼
papermind/main.py          ← 入口，启动 LangGraph pipeline
    │
    ▼
orchestrator/graph.py      ← 构建 StateGraph，所有节点通过 Command(goto=...) 跳转
    │
    ├── 主 Agent (MainAgent)     ← 单一 LLM 实例，有对话记忆，负责决策
    └── Sub-Agents (×6)          ← 无状态执行单元，各自独立完成具体任务
```

---

## LangGraph 节点流转图

```
explore_directions
    │
    ▼
synthesize_outline
    │
    ▼
research_sections ◄──────────────────────────────┐
    │                                             │
    ▼                                             │
check_coverage ──── sufficient=False, count<2 ───┘
    │
    │ sufficient=True 或 count>=2
    ▼
build_index
    │
    ▼
write_sections
    │
    ▼
review_sections ◄────────────────────────────────┐
    │                                             │
    │ has_issues AND revision_count < max_revisions
    ▼                                             │
fix_issues ──────────────────────────────────────┘
    │
    │ no_issues OR revision_count >= max_revisions
    ▼
polish_sections
    │
    ▼
check_consistency
    │
    ▼
merge_final
    │
    ▼
  END  →  runs/{timestamp}/survey.md
```

**分支说明：**

| 节点 | 条件 | 跳转目标 |
|---|---|---|
| `check_coverage` | sufficient=True 或已检查 ≥2 次 | `build_index` |
| `check_coverage` | sufficient=False 且 count<2 | `research_sections`（补充研究） |
| `review_sections` | 有 issues 且 revision_count < max_revisions | `fix_issues` |
| `review_sections` | 无 issues 或 revision_count ≥ max_revisions | `polish_sections` |
| `fix_issues` | 始终 | `review_sections`（重新审核） |

---

## 主 Agent（MainAgent）

**文件：** `src/orchestrator/main_agent.py`

**定位：** 决策者，不执行具体搜索或写作，只做规划和判断。

**上下文：**
- 系统提示词：`MAIN_AGENT_SYSTEM`（描述三阶段工作流、输出格式、工作原则）
- 对话历史：`state["agent_messages"]`，跨节点累积，每次调用都带完整历史
- 存储格式：`[{"role": "human"|"ai", "content": "..."}]`，存在 LangGraph state 中

**被调用的节点：**

| 节点 | 调用方式 | 输入 | 输出 |
|---|---|---|---|
| `explore_directions` | `chat()` | 研究课题 | 3-5 个子方向（纯文本，每行一个） |
| `synthesize_outline` | `chat_json()` | Explorer 探索结果摘要 | `{recommended_concurrency, chapters:[{title, description, sub_questions:[]}]}` |
| `check_coverage` | `chat_json()` | 大纲 + 已收集文献数量 | `{sufficient, weak_areas:[], reason}` |

**日志：** `runs/{run}/data/main_agent.json`（每轮对话追加一条）

---

## Sub-Agents

所有 Sub-Agent 继承 `SubAgentBase`，统一入口 `run(task)` 提供：
- `asyncio.wait_for` 超时保护
- 异常捕获，失败返回 `status="failed"`
- I/O 日志写入 `runs/{run}/data/{agent_id}.json`

### ExplorerAgent

**文件：** `src/agents/explorer.py`  
**触发：** `explore_directions` 节点，每个子方向一个实例，并发数 = `min(3, agent_concurrency)`  
**超时：** `EXPLORER_TIMEOUT`（默认 600s）

**上下文（每次迭代）：**
- `_plan_chain`：系统提示 + 研究方向 + 已搜索摘要 → 本轮搜索词（4-6 个）
- `_analyze_chain`：系统提示 + 搜索结果 → `{sufficient, reason, missing_aspects}`
- `_synthesize_chain`：系统提示 + 全部搜索结果 → 结构化报告

**内部循环（最多 `max_iterations=4` 轮）：**
```
plan_queries → ddg_search_batch → analyze
    │
    ├── sufficient=True → 跳出循环
    └── sufficient=False → 继续下一轮
    │
    ▼（循环结束后）
synthesize → 返回 AgentResult(metadata=[report])
```

**输出结构：**
```json
{
  "direction": "...",
  "mainstream_methods": ["...", "..."],
  "key_controversies": ["..."],
  "representative_papers": [{"title","authors","year","venue","significance"}],
  "recent_trends": "...",
  "summary": "..."
}
```

---

### ResearcherAgent

**文件：** `src/agents/researcher.py`  
**触发：** `research_sections` 节点，每个 `sub_question` 一个实例，并发数 = `min(3, agent_concurrency)`  
**超时：** `RESEARCHER_TIMEOUT`（默认 900s）

**上下文（每次迭代）：**
- `_plan_chain`：系统提示 + 研究问题 + 已收集论文标题 → 本轮搜索词（3-5 个）
- `_extract_chain`：系统提示 + 搜索结果 + 已有标题 → 论文元数据 JSON 数组

**内部循环（最多 `max_iterations=8` 轮）：**
```
plan_queries → ddg_search_batch → extract_papers → upsert to SQLite
    │
    ├── new_count=0 连续 2 轮 → 停止
    └── 否则继续
```

**输出：** `AgentResult(papers=[PaperRecord...], metadata=[{paper_id, title}...])`

---

### WriterAgent

**文件：** `src/agents/writer.py`  
**触发：** `write_sections` 和 `fix_issues` 节点，每个章节一个实例，并发数 = `min(3, agent_concurrency)`  
**超时：** `WRITE_TIMEOUT`（默认 1800s）

**输入（task.instruction JSON）：**
```json
{"section_title": "...", "outline_text": "...", "adjacent_context": "前后章节各500字"}
```

**上下文（每次尝试）：**
- `_write_chain`：系统提示 + 章节大纲 + RAG 检索片段 + 上轮反馈（如有）→ 章节草稿
- `_verify_chain`：系统提示 + 草稿 + RAG 检索片段 → `PASS` 或 `FAIL + 问题列表`
- RAG：`Retriever.dual_search()`（Path A 直接向量搜索 + Path B LLM 关键词扩展搜索）

**内部循环（最多 `writer_max_retries=3` 次）：**
```
dual_search → write → verify
    │
    ├── PASS → 返回草稿
    └── FAIL → feedback 注入下一轮 write
```

**输出：** `AgentResult(draft_text="...")`

---

### ReviewerAgent

**文件：** `src/agents/reviewer.py`  
**触发：** `review_sections` 节点，每个章节一个实例，并发数 = `min(3, agent_concurrency)`  
**超时：** `REVIEWER_TIMEOUT`（默认 300s）

**输入（task.instruction JSON）：**
```json
{"section_title": "...", "draft_text": "..."}
```

**上下文：**
- RAG：`Retriever.search(draft[:600], top_k=8)`（简单向量搜索，不做关键词扩展）
- `_chain`：系统提示 + 章节标题 + 草稿 + RAG 片段 → `{passed, issues:[]}`

**输出：** `AgentResult(issues=[...])` — issues 为空则 status="success"，否则 status="partial"

---

### PolisherAgent

**文件：** `src/agents/polisher.py`  
**触发：** `polish_sections` 节点，每个章节一个实例，并发数 = `min(3, agent_concurrency)`  
**超时：** `POLISHER_TIMEOUT`（默认 300s）

**输入（task.instruction JSON）：**
```json
{"section_title": "...", "draft_text": "..."}
```

**上下文：**
- `_chain`：系统提示 + 草稿 → 润色后正文（保持引用标注不变）

**输出：** `AgentResult(polished_text="...")`

---

### ConsistencyCheckerAgent

**文件：** `src/agents/polisher.py`  
**触发：** `check_consistency` 节点，串行，只有一个实例  
**超时：** `POLISHER_TIMEOUT`（默认 300s）

**输入（task.instruction JSON）：**
```json
{"sections": [{"title":"...", "head":"前300字", "tail":"后300字"}]}
```

**上下文：**
- `_chain`：系统提示 + 各章节首尾摘要（总量 ≤5000 字）→ 一致性报告

**输出：**
```json
{
  "terminology_issues": [{"term","variants":[],"recommended"}],
  "transition_issues": ["章节X与Y衔接问题描述"],
  "citation_issues": ["引用格式异常描述"]
}
```

---

## merge_final（无 LLM，纯程序化）

**触发：** `check_consistency` 之后，串行  
**操作：**
1. 按大纲顺序拼接 `polished_sections`
2. 执行术语替换（字符串替换，基于 `terminology_issues`）
3. 对 `transition_issues` 标记的相邻章节，调用轻量 LLM 生成一句过渡句插入章节开头
4. 正则替换 `[Author et al., Year]` → `[1]`、`[2]`... 并生成 References 列表
5. 写入 `runs/{run}/survey.md`

---

## LangGraph State 字段总览

```python
# 输入
research_topic: str
run_dir: str
db_path: str
target_papers: int

# 主 Agent 记忆
agent_messages: list[dict]        # [{"role":"human"|"ai","content":"..."}]

# Phase 1
sub_directions: list[str]         # 3-5 个子方向
explorer_results: list[dict]      # 每个方向的 Explorer 报告

# Phase 2
research_outline: list[dict]      # [{title, description, sub_questions:[]}]
agent_concurrency: int            # LLM 建议的并发数（1-3，硬上限 3）

# Phase 3
researcher_results: dict          # sub_question → AgentResult.model_dump()
coverage_ok: bool
coverage_check_count: int         # 最多循环 2 次

# Phase 4
faiss_built: bool
section_drafts: dict              # section_key → draft_text

# Phase 5
review_issues: dict               # section_key → [issue_str]
revision_count: int               # 当前修订轮次
max_revisions: int                # 默认 2

# Phase 6
polished_sections: dict           # section_key → polished_text
consistency_report: dict          # ConsistencyChecker 输出
final_output: str                 # 最终 Markdown 全文
output_path: str                  # survey.md 路径
```

---

## 并发控制

- 所有并发节点使用 `asyncio.Semaphore(min(3, state["agent_concurrency"]))`
- `agent_concurrency` 由主 Agent 在 `synthesize_outline` 阶段根据课题复杂度决定（1-3）
- 硬上限：代码层面 `min(3, ...)` 保证不超过 3
- 方向数量上限：`explore_directions` 截断为最多 5 个

## 输出文件结构

```
runs/{timestamp}-{slug}/
├── papers.db          ← SQLite，所有论文元数据
├── pdfs/              ← 下载的 PDF 文件
├── faiss.index        ← FAISS 向量索引
├── chunks.pkl         ← 对应的 Chunk 列表
├── survey.md          ← 最终综述
└── data/
    ├── main_agent.json           ← 主 Agent 每轮对话记录
    ├── explorer-{dir}.json       ← 各 ExplorerAgent I/O
    ├── researcher-{sq}.json      ← 各 ResearcherAgent I/O
    ├── writer-{key}.json         ← 各 WriterAgent I/O
    ├── reviewer-{key}.json       ← 各 ReviewerAgent I/O
    ├── polisher-{key}.json       ← 各 PolisherAgent I/O
    └── consistency-checker.json  ← ConsistencyChecker I/O
```
