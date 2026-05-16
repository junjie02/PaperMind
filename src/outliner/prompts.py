"""Prompt strings for the outline-drafting LangGraph nodes."""


WRITER_SYSTEM_PROMPT = """你是一位严谨的文献综述写作者。要求：
- 语言客观、紧贴论文给出的事实，不要编造未在材料中出现的研究内容。
- 每段描述中要穿插文献引用，参考文献综述风格，多个引用之间用 " | " 分隔，例如：
  `... 这一方法在视觉任务上取得突破 [Vision Transformer | Swin Transformer],H等人实现了...提升8%[GLU aTn Transformer]`。
- 引用使用论文**原标题字符串**（保持英文原文），不要使用 ID、数字编号或缩写。
- 输出 Markdown，章节用 `# 标题`，子章节用 `## 标题`，描述紧跟在标题下一行。
"""


CLUSTER_PROMPT = """请把下列论文按子主题归到 {min_groups}-{max_groups} 组中，组数你自己决定。

研究课题：{topic}

论文列表（共 {n_papers} 篇）：
{paper_lines}

要求：
- 每组要语义内聚，组与组之间区分明显。
- 给每组起一个简洁的中文子主题名（10-30 字）。
- 一篇论文可以出现在多组里。

只输出一个 JSON 对象，结构如下，不要任何额外文字：
{{
  "groups": [
    {{"name": "中文子主题名", "paper_ids": ["arxiv:1234.56789", "openreview:abcd1234"]}}
  ]
}}
"""


PLAN_CHAPTERS_PROMPT = """你是文献综述结构规划专家。以下是一篇关于「{topic}」的综述中所有论文，已按子主题分组：

{groups_block}

请规划一份综述大纲的章节标题列表，要求：
- 必须包含 Introduction 和 Conclusion（固定首尾，subsections 留空列表）
- 中间章节数量约 {n_sections} 个，每个章节对应一个或多个子主题
- 章节标题要能把这个领域的工作分门别类讲清楚，体现逻辑递进
- 必须有专门讨论开放问题或未来研究方向的章节
- 每个中间章节规划 {n_subsections} 个子节标题，子节标题要具体、有区分度
- `directions` 字段必须使用上方分组的**原始子主题名**，不要改写或翻译
- 只输出标题列表，不要写内容

只输出一个 JSON 对象，不要任何额外文字：
{{
  "title": "综述的正式学术标题（中文或英文均可，10-30字）",
  "chapters": [
    {{
      "title": "章节标题",
      "directions": ["子主题名1", "子主题名2"],
      "subsections": ["子节标题1", "子节标题2"]
    }}
  ]
}}
"""


DRAFT_OUTLINE_PROMPT = """请为以下章节起草详细大纲片段。

整体研究课题：{topic}
本章节标题：{chapter_title}
子节标题列表：
{subsections_block}

本章节对应的论文（按相关性排序）：

{paper_blocks}

起草要求：
- 可在原章节标题基础上优化措辞（用 `# 标题` 输出）
- 严格按照上方子节标题列表输出子节（`## 标题`），可微调措辞但不要增减
- 每个子节标题下几句已知文献描述在描述句尾引用文献，并说明该子节涵盖哪些工作与结论。
- 每句描述若有引用文献，描述结尾必须按 system prompt 规则给出 `[Title | Title]` 引用。务必保证所有参考文献都引用到。
- 输出 Markdown，`#` 为章节，`##` 为子节
"""


MERGE_OUTLINES_PROMPT = """请把下面 {n_groups} 份子大纲合并为一份完整综述大纲。

整体研究课题：{topic}

子大纲列表（用 `---` 分隔）：

{outlines_joined}

合并要求：
- 添加 `# Introduction` 开头和 `# Conclusion` 结尾。
- 合并语义相近的章节，避免重复，但不要丢失信息；
- 严格保留子大纲里的 `[Title | Title]` 引用位置不变；合并后若同一句话指向多篇引用，把引用合并成一个方括号内用 ` | ` 分隔。
- 输出 Markdown，不要解释，不要前后多余文字。
- 如果只有 1 份子大纲，只做小幅润色 + 补 Introduction / Conclusion 即可。
"""


REVIEWER_SYSTEM_PROMPT = """你是一位严格的学术综述评审专家。你的任务是评估文献综述大纲的质量。

评审标准（共 3 条）：
1. 组织结构与主题的契合度：章节划分是否合理、逻辑是否清晰、是否真的围绕综述主题展开。
2. 是否有清晰且有价值的贡献：大纲要能体现出这篇综述能给读者带来什么新的认识或整合，不能只是罗列论文；必须有专门讨论开放问题或后续研究方向的部分。

输出格式为 JSON，key 固定如下，不要输出任何额外文字：
{{
  "THOUGHTS": "逐条分析三个评审标准，chain-of-thought 风格，每个章节思考后输出 【章节评审】yes 或者 【章节评审】no",
  "PER_CRITERIA": ["yes 或 no", "yes 或 no", "yes 或 no"],
  "SUFFICIENT": "yes 或 no（三条标准均满足则 yes，否则 no）",
  "FEEDBACK": "SUFFICIENT 为 no 时填写具体修改意见；否则留空字符串"
}}
"""


REVIEW_OUTLINE_PROMPT = """请评审以下文献综述大纲。

研究课题：{topic}
当前修订轮次：第 {revision_round} 轮

大纲内容：
---
{outline}
---

可用论文列表（共 {n_papers} 篇）：
{paper_titles}
"""


REVISE_OUTLINE_PROMPT = """请根据评审反馈修订以下文献综述大纲。

研究课题：{topic}
目标章节数：{n_sections}
每章节子节数：{n_subsections}

当前大纲：
---
{current_outline}
---

评审反馈：
{feedback}

修订要求：
- 针对评审指出的问题逐一改进。
- **严禁删除任何 `[Title | Title]` 引用**，只能新增或调整位置。
- 必须有 `# Introduction` 和 `# Conclusion`。
- 输出完整修订后的 Markdown 大纲，不要解释。
"""
