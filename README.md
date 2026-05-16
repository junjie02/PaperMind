# PaperMind

自动化文献综述生成工具。主 Agent（LangGraph）编排多个并发 Sub-Agent，完成从方向探索、论文挖掘、写作、评审到润色的全流程。

## 架构

```
PM "topic" -n 30
     │
     ▼
主 Agent (LangGraph + MiniMax)
     │
     ├── Phase 1: Explorer Sub-Agent × 3-5  ──→ DDG 搜索 + 网页抓取
     │             （背景调研，3轮迭代，自主决定搜索/抓取/结束）
     │
     ├── Phase 2: 主 Agent 生成大纲 + 分配文献数量
     │             （输出 outline.md，按重要性为每个 sub_question 分配目标篇数）
     │
     ├── Phase 3: Researcher Sub-Agent × N  ──→ DDG 搜索 + 网页抓取 + PDF 下载
     │             （针对性收集论文，达到目标数量即停止，最多12轮）
     │
     ├── Phase 4: Build Index ──→ PDF 提取 / 元数据 fallback → FAISS 向量索引
     │
     ├── Phase 5: Writer Sub-Agent × N      ──→ RAG 检索（上下文窗口 + 随机探索）
     │             （迭代写作，每轮多查询并行检索，注入可用论文列表）
     │
     ├── Phase 6: Reviewer Sub-Agent × N    ──→ RAG 逐句核查
     │             （迭代验证，每轮批量检索带引用的论断）
     │
     ├── Phase 7: Polisher + Consistency    ──→ 逐节润色 + 全局一致性检查
     │
     ├── Phase 8: Final Review              ──→ 终审 LLM 全局修正
     │             （整篇综述 + 论文标题列表发给独立模型做最终审校）
     │
     └── Phase 9: Merge Final               ──→ 引用编号 + GB/T 7714 参考文献 + 输出
```

- **搜索工具**：DDG 搜索 + trafilatura 网页抓取（`src/mcp_servers/ddg_search.py`）
- **RAG 检索**：FAISS + sentence-transformers + 上下文窗口 + 随机跳跃（`src/rag/retriever.py`）
- **LLM**：MiniMax（OpenAI 兼容 API），所有 Agent 共用；终审可配置独立模型
- **向量模型**：`all-MiniLM-L6-v2`
- **引用匹配**：embedding 相似度匹配（阈值 0.75）

## 安装

```bash
python -m venv venv
source venv/bin/activate
pip install -e .
```

## 配置

```bash
cp .env.example .env
# 编辑 .env，填入 OPENAI_API_KEY 和代理地址
```

关键环境变量：

| 变量 | 说明 |
|------|------|
| `OPENAI_API_KEY` | MiniMax / DeepSeek / OpenAI API Key |
| `OPENAI_BASE_URL` | API 地址（默认 MiniMax） |
| `OPENAI_MODEL` | 模型名称（默认 `MiniMax-M2.7`） |
| `REVIEW_API_KEY` | 终审 LLM API Key（可选，不填用主 LLM） |
| `REVIEW_BASE_URL` | 终审 LLM API 地址 |
| `REVIEW_MODEL` | 终审 LLM 模型名称 |
| `HTTP_PROXY` | DDG 搜索和 PDF 下载代理 |
| `MAX_CONCURRENT_AGENTS` | 最大并发 Sub-Agent 数（默认 4，实际由主 Agent 决定，硬上限 3） |

## 使用

```bash
# 生成文献综述
PM "agent 安全研究" -n 15
PM "diffusion models for video" -n 30

# 指定输出目录
PM "vision transformers" --out runs/my-run

# 从已有运行目录的某个阶段恢复
PM "LLM 推理加速" --out runs/existing-run --resume build_index
PM "LLM 推理加速" --out runs/existing-run --resume write_sections

# 跳过 reviewer 评审
PM "topic" -n 20 --skip-review

# 自动生成 PDF
PM "topic" -n 20 --pdf

# 单独将 md 转为 PDF
PM --to-pdf runs/existing-run/survey.md

# 开启 DEBUG 日志
PM "topic" -v

# 运行评测
cd src && python -m evaluate "../runs/existing-run"
```

可用的恢复阶段：`explore_directions`、`synthesize_outline`、`research_sections`、`check_coverage`、`build_index`、`write_sections`、`polish_sections`、`check_consistency`、`final_review`、`merge_final`

## 输出目录结构

```
runs/{timestamp}-{slug}/
├── papers.db              # 论文元数据（SQLite）
├── pdfs/                  # PDF 文件 + 无 PDF 论文的 MD 元数据文件
├── faiss.index            # FAISS 向量索引
├── chunks.pkl             # 文本分块缓存
├── outline.md             # 研究大纲（含每个问题的目标文献数）
├── sections/              # 各 sub_question 的独立 md 文件
├── draft.md               # 合并后的正文草稿
├── data/                  # 完整 Agent I/O 日志（JSON）
│   ├── main_agent.json
│   ├── explorer-*_iterations.json
│   ├── researcher-*_iterations.json
│   ├── writer-*_iterations.json
│   ├── reviewer-*_iterations.json
│   └── ...
├── survey.md              # 最终综述
├── survey.pdf             # PDF 导出（--pdf）
└── evaluation_report.json # 评测报告
```

## 目录结构

```
src/
├── papermind/       # 入口（PM 命令 + PDF 导出）
├── shared/          # 共享模型、数据库、配置、LLM 工厂
├── mcp_servers/     # DDG 搜索 + 网页抓取 + RAG 检索
├── agents/          # Explorer / Researcher / Writer / Reviewer / Polisher
├── orchestrator/    # LangGraph 主 Agent 编排
├── rag/             # FAISS 索引、分块、检索（含上下文窗口）
└── evaluate/        # 评测指标脚本
```
