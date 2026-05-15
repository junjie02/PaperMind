"""Prompt strings for the outline-drafting LangGraph nodes."""


WRITER_SYSTEM_PROMPT = """你是一位严谨的文献综述写作者。要求：
- 语言客观、紧贴论文给出的事实，不要编造未在材料中出现的研究内容。
- 每段描述末尾必须用方括号标注引用，多个引用之间用 " | " 分隔，例如：
  `... 这一方法在视觉任务上取得突破 [Vision Transformer | Swin Transformer]`。
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
- 每篇论文必须且仅出现在一组里。

只输出一个 JSON 对象，结构如下，不要任何额外文字：
{{
  "groups": [
    {{"name": "中文子主题名", "paper_ids": ["arxiv:1234.56789", "openreview:abcd1234"]}}
  ]
}}
"""


DRAFT_OUTLINE_PROMPT = """请为以下子主题起草一份综述大纲片段。

整体研究课题：{topic}
当前子主题：{direction}
目标章节数：{n_sections}
每章节子节数：{n_subsections}

下面是属于本子主题的论文（按相关性排序）：

{paper_blocks}

输出要求：
- Markdown 格式，每个章节用 `# 标题`，子节用 `## 标题`。
- 每个标题下面写 1-3 句中文描述，结尾必须按 system prompt 规则给出 `[Title | Title]` 引用。
- 不要写 Introduction / Conclusion——这是子大纲，最终合并阶段会补全。
- 控制在 {n_sections} 个 `#` 章节左右；每个章节带 {n_subsections} 个 `##` 子节。
"""


MERGE_OUTLINES_PROMPT = """请把下面 {n_groups} 份子大纲合并为一份完整综述大纲。

整体研究课题：{topic}
目标章节数：{n_sections}
每章节子节数：{n_subsections}

子大纲列表（用 `---` 分隔）：

{outlines_joined}

合并要求：
- 必须有 `# Introduction` 开头和 `# Conclusion` 结尾。
- 合并语义相近的章节，避免重复；最终 `#` 章节数尽量贴近 {n_sections}。
- 保留子大纲里的 `[Title | Title]` 引用语法不变；同一段如果合并多组，把引用合并成一个方括号内用 ` | ` 分隔。
- 输出 Markdown，不要解释，不要前后多余文字。
- 如果只有 1 份子大纲，只做小幅润色 + 补 Introduction / Conclusion 即可。
"""


REVIEWER_SYSTEM_PROMPT = """你是一位严格的学术综述评审专家。你的任务是评估文献综述大纲的质量。
评审标准：
1. 结构完整性：是否有 Introduction 和 Conclusion？章节层级是否清晰？
2. 覆盖度：是否覆盖了研究课题的主要子方向？有无明显遗漏？
3. 引用质量：每个章节是否有 [Title] 引用支撑？引用是否与描述内容匹配？
4. 逻辑连贯：章节之间是否有逻辑递进关系？是否存在重复或矛盾？
5. 描述质量：描述是否客观、具体、紧贴论文内容？有无空泛或编造的表述？
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

请按以下 JSON 格式输出评审结果，不要有任何额外文字：
{{
  "approved": true/false,
  "score": 1-10,
  "strengths": ["优点1", "优点2"],
  "weaknesses": ["问题1", "问题2"],
  "suggestions": ["修改建议1", "修改建议2"],
  "summary": "一句话总结评审意见"
}}
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
- 评分：{score}/10
- 问题：{weaknesses}
- 建议：{suggestions}

修订要求：
- 针对评审指出的问题逐一改进。
- 保持 `[Title | Title]` 引用语法不变。
- 必须有 `# Introduction` 和 `# Conclusion`。
- 输出完整修订后的 Markdown 大纲，不要解释。
"""
