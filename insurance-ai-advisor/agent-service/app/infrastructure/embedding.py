"""本地向量模型：把文本变成向量。

用的是 Qwen3-Embedding-0.6B：
  - 输出 1024 维向量，必须和 Milvus collection 的维度一致
  - 支持 32K 上下文，保险条款整条塞得下
  - 中文检索能力强，0.6B 参数在 CPU 上也能跑
"""

import logging
from functools import lru_cache

from sentence_transformers import SentenceTransformer

from app.core.config import settings

logger = logging.getLogger(__name__)

# Qwen3-Embedding 的查询侧需要加任务指令，文档侧不加——这是非对称检索的通行做法。
# 提示词直接用模型自带的（配置在 config_sentence_transformers.json 的 prompts.query 里）：
#   "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"
# 不要自己手写替代品，模型是按官方这句训练的。
QUERY_PROMPT_NAME = "query"


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """加载模型。lru_cache 保证整个进程只加载一次。

    首次调用约 100 秒（大头是导入 torch），之后每次都是瞬间返回。
    """
    logger.info(
        "正在加载向量模型：%s（设备 %s）", settings.embedding_model_path, settings.embedding_device
    )
    model = SentenceTransformer(settings.embedding_model_path, device=settings.embedding_device)
    logger.info("向量模型加载完成，最大序列长度 %s", model.max_seq_length)
    return model


def encode_texts(
    texts: list[str], *, is_query: bool = False, batch_size: int | None = None
) -> list[list[float]]:
    """把文本编码成向量。

    normalize_embeddings=True 做归一化，这样余弦相似度就等于点积，
    Milvus 用 COSINE 度量时结果才和本地计算一致。
    """
    if not texts:
        return []

    model = get_embedding_model()
    vectors = model.encode(
        texts,
        batch_size=batch_size or settings.embedding_batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
        prompt_name=QUERY_PROMPT_NAME if is_query else None,
    )
    return [vector.tolist() for vector in vectors]
