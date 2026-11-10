"""重排模型：对召回的候选做精排。

为什么两路召回之后还要重排：
  向量检索和 BM25 都是"先粗筛"——它们各自独立判断相关性，
  用的是双塔结构（问题和文档分别编码），问题与文档之间没有交互。
  而 reranker 是交叉编码器（cross-encoder），把"问题 + 文档"拼在一起送进模型，
  能捕捉两者之间的细粒度关联，精度高得多——代价是慢，所以要先用召回把候选压到几十条。
"""

import logging
from functools import lru_cache

from sentence_transformers import CrossEncoder

from app.core.config import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    logger.info(
        "正在加载重排模型：%s（设备 %s）", settings.reranker_model_path, settings.embedding_device
    )
    model = CrossEncoder(
        settings.reranker_model_path,
        device=settings.embedding_device,
        max_length=settings.reranker_max_length,
    )
    logger.info("重排模型加载完成")
    return model


def rerank_scores(query: str, documents: list[str]) -> list[float]:
    """给每个 (查询, 文档) 组合打一个相关性分数，分数越高越相关。

    bge-reranker 系列输出的是 logits，没有固定取值范围，
    只用于排序，不要当成概率解读。
    """
    if not documents:
        return []
    pairs = [(query, doc) for doc in documents]
    # 关掉进度条，否则每打一次分就往日志里塞一行 "Batches: 100%|..."
    scores = get_reranker().predict(pairs, show_progress_bar=False)
    return [float(score) for score in scores]
