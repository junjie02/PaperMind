"""Prompts for DeepSeek-based section writing and fact verification."""

WRITE_SYSTEM_PROMPT = "你是一位严谨的学术文献综述写作者。你只能基于提供的论文片段进行写作，不得编造任何内容。"

WRITE_USER_PROMPT = """\
## 章节大纲

{section_outline}

## 相关论文片段

{chunks_text}

{feedback_block}\
## 写作要求

- 严格基于上方论文片段，每个事实性论断必须有引用，格式为 `[Author et al., Year]`
- 不得编造片段中未出现的内容；若片段不足以支撑某个子节，可简短说明"相关研究有限"
- 学术综述体，段落式论述（非列表），注重论文间的对比、联系和发展脉络
- 字数：300-600 字
- 直接输出章节正文（Markdown），不要加任何额外说明或前言
"""

FEEDBACK_BLOCK_TEMPLATE = """\
## 上轮审核反馈

{feedback}

请根据以上反馈修正后重写本章节。

"""

VERIFY_SYSTEM_PROMPT = "你是一位文献综述事实核查员。你的任务是检查综述章节中的每个论断是否有论文片段支撑。"

VERIFY_USER_PROMPT = """\
## 待审核章节

{draft}

## 可供核查的论文片段

{verify_chunks_text}

## 核查任务

逐句检查章节中每个事实性论断（含引用标注）是否能在上方片段中找到依据。

- 若全部有据可查：**只回复** `PASS`
- 若有无法核实的论断：**只回复** `FAIL`，然后换行列出具体问题（每条一行），例如：
  FAIL
  - 第2段提到"X方法达到90%准确率"，片段中未见此数据
  - 引用[Smith et al., 2023]的结论与片段内容不符

只回复 PASS 或 FAIL 开头的内容，不要其他说明。
"""

KEYWORD_EXPANSION_PROMPT = """\
给定文献综述章节："{section_title}"
大纲描述：{outline_text}

列出 5-8 个用于检索相关论文的关键短语（英文或中文均可），每行一个，不要编号。\
"""
