"""文本切分：把长文本切成适合向量化的片段。

为什么要切分而不是整条条款直接向量化：
  - 一次检索要能定位到"具体哪一句话"，切片太长会稀释语义
  - 相邻切片留重叠，避免关键信息正好被切断在两片之间

本项目的数据特点是"一条条款就是一个语义单元"，多数条款本身就短于 chunk_size，
所以大部分条款会切成 1 片——这是正确行为，切分逻辑是为将来更长的条款准备的。
"""

import re

# 中文句末标点。用 (?<=...) 零宽断言，保留标点本身
_SENTENCE_END = re.compile(r"(?<=[。！？；!?;])")


def split_sentences(text: str) -> list[str]:
    """按换行和句末标点切句。"""
    sentences: list[str] = []
    for block in text.split("\n"):
        block = block.strip()
        if not block:
            continue
        sentences.extend(part for part in _SENTENCE_END.split(block) if part.strip())
    return sentences


def split_text(text: str, chunk_size: int = 500, overlap: int = 80) -> list[str]:
    """把文本切成若干片段。

    策略：先切句，再贪心地合并句子直到接近 chunk_size；
    开新片段时带上上一片末尾的 overlap 个字符作为重叠。
    """
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    # 防御：重叠不能大于等于切片长度，否则下面的循环会原地打转
    if overlap >= chunk_size:
        overlap = chunk_size // 5

    chunks: list[str] = []
    current = ""

    for sentence in split_sentences(text):
        if len(current) + len(sentence) <= chunk_size:
            current += sentence
            continue

        if current:
            chunks.append(current)
            tail = current[-overlap:] if overlap > 0 else ""
        else:
            tail = ""

        current = tail + sentence

        # 单句本身就超过 chunk_size 时硬切（中文条款很少见，但要有兜底）
        while len(current) > chunk_size:
            chunks.append(current[:chunk_size])
            current = current[chunk_size - overlap :]

    if current.strip():
        chunks.append(current)

    return [c for c in chunks if c.strip()]
