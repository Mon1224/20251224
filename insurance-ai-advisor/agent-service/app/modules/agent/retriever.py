"""条款检索：稠密 + 稀疏双路召回，RRF 融合，再重排。

三段式的理由：
  1. 召回（两路并行）—— 目标是"别漏"，宁可多召回一些
       稠密：按语义相似找，能处理"后悔了能退吗"→"犹豫期"这种同义改写
       稀疏：按关键词找，能处理"第 3.2 条""如实告知"这种精确匹配
  2. 融合（RRF）—— 把两份排名合并成一份，且不需要校正两边的分数量纲
  3. 重排（reranker）—— 目标变成"别错"，用交叉编码器精排出最终 Top-N
"""

import asyncio
import logging
from dataclasses import dataclass, field

from app.core.config import settings
from app.infrastructure.embedding import encode_texts
from app.infrastructure.milvus import get_milvus_client
from app.infrastructure.reranker import rerank_scores
from app.modules.agent.schemas import ClauseHit, ClauseSearchOut

logger = logging.getLogger(__name__)

# 从 Milvus 取回的字段。clause_no / title / product_name 是冗余存储的，
# 为的是检索命中后不用再回 PostgreSQL 查一次。
OUTPUT_FIELDS = [
    "clause_id",
    "product_id",
    "chunk_index",
    "clause_no",
    "title",
    "product_name",
    "content",
]


@dataclass
class Candidate:
    chunk_id: int
    clause_id: int
    product_id: int
    clause_no: str
    title: str
    product_name: str
    content: str
    dense_rank: int | None = None
    dense_score: float | None = None
    sparse_rank: int | None = None
    sparse_score: float | None = None
    rrf_score: float = 0.0
    rerank_score: float | None = None
    extra: dict = field(default_factory=dict)


def _to_candidates(raw_hits, *, path: str) -> list[Candidate]:
    """把 Milvus 原始返回转成 Candidate，并记录它在这一路的名次与分数。"""
    candidates: list[Candidate] = []
    for rank, hit in enumerate(raw_hits, start=1):
        entity = hit["entity"]
        candidate = Candidate(
            chunk_id=int(hit["id"]),
            clause_id=int(entity["clause_id"]),
            product_id=int(entity["product_id"]),
            clause_no=str(entity["clause_no"]),
            title=str(entity["title"]),
            product_name=str(entity["product_name"]),
            content=str(entity["content"]),
        )
        if path == "dense":
            candidate.dense_rank = rank
            candidate.dense_score = float(hit["distance"])
        else:
            candidate.sparse_rank = rank
            candidate.sparse_score = float(hit["distance"])
        candidates.append(candidate)
    return candidates


def rrf_fuse(
    dense: list[Candidate], sparse: list[Candidate], *, k: int = 60
) -> list[Candidate]:
    """Reciprocal Rank Fusion：按名次融合两路结果。

    公式：score(d) = Σ 1 / (k + rank_i(d))，k 是平滑常数（常用 60）。

    为什么用 RRF 而不是把两边的分数加权相加：
      余弦相似度在 0~1 之间，BM25 分数可能到十几，两个量纲根本没法直接相加。
      加权的话，"权重调多少"完全靠试，换个查询就失效。
      RRF 只看名次不看分数，天然免疫量纲问题——这是它最大的工程价值。
    """
    merged: dict[int, Candidate] = {}
    scores: dict[int, float] = {}

    for results in (dense, sparse):
        for rank, candidate in enumerate(results, start=1):
            key = candidate.chunk_id
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            if key in merged:
                # 两路都召回了它，把两边的排名信息合并到同一个对象上
                existing = merged[key]
                existing.dense_rank = existing.dense_rank or candidate.dense_rank
                existing.dense_score = existing.dense_score or candidate.dense_score
                existing.sparse_rank = existing.sparse_rank or candidate.sparse_rank
                existing.sparse_score = existing.sparse_score or candidate.sparse_score
            else:
                merged[key] = candidate

    for key, candidate in merged.items():
        candidate.rrf_score = scores[key]

    return sorted(merged.values(), key=lambda c: c.rrf_score, reverse=True)


def search_clauses(
    query: str,
    *,
    mode: str = "hybrid",
    top_k: int | None = None,
    final_top_n: int | None = None,
    use_rerank: bool | None = None,
    product_id: int | None = None,
    client=None,
) -> ClauseSearchOut:
    """条款检索主入口。mode 用来做效果对比：hybrid / dense / sparse。"""
    query = query.strip()
    if not query:
        raise ValueError("查询内容不能为空")

    top_k = top_k or settings.retrieval_top_k
    final_top_n = final_top_n or settings.retrieval_final_top_n
    use_rerank = settings.enable_rerank if use_rerank is None else use_rerank

    client = client or get_milvus_client()
    collection = settings.milvus_collection
    # 需要限定产品时，用 Milvus 的标量过滤，比事后在 Python 里筛更高效
    expr = f"product_id == {product_id}" if product_id else None

    dense_candidates: list[Candidate] = []
    sparse_candidates: list[Candidate] = []

    if mode in ("hybrid", "dense"):
        vector = encode_texts([query], is_query=True)[0]
        raw = client.search(
            collection_name=collection,
            data=[vector],
            anns_field="dense",
            limit=top_k,
            output_fields=OUTPUT_FIELDS,
            filter=expr,
        )[0]
        dense_candidates = _to_candidates(raw, path="dense")

    if mode in ("hybrid", "sparse"):
        # 稀疏检索直接传原始文本：Milvus 会用 BM25 函数把它转成稀疏向量。
        # 这是用了 BM25 Function 之后才有的便利，普通稀疏字段得自己算好再传。
        raw = client.search(
            collection_name=collection,
            data=[query],
            anns_field="sparse",
            limit=top_k,
            output_fields=OUTPUT_FIELDS,
            filter=expr,
        )[0]
        sparse_candidates = _to_candidates(raw, path="sparse")

    if mode == "dense":
        fused = dense_candidates
        for rank, candidate in enumerate(fused, start=1):
            candidate.rrf_score = 1.0 / (settings.retrieval_rrf_k + rank)
    elif mode == "sparse":
        fused = sparse_candidates
        for rank, candidate in enumerate(fused, start=1):
            candidate.rrf_score = 1.0 / (settings.retrieval_rrf_k + rank)
    else:
        fused = rrf_fuse(dense_candidates, sparse_candidates, k=settings.retrieval_rrf_k)

    merged_count = len(fused)

    # 先截断再重排：reranker 是整条链路上最慢的一环，不能把 40 条都丢给它。
    # 实测在纯 CPU 上，bge-reranker-v2-m3 每对约 1.5 秒，10 条就是 15 秒——
    # 这是本项目最大的性能瓶颈，参数调大要谨慎。
    pool_size = max(final_top_n, settings.retrieval_rerank_pool)
    pool = fused[:pool_size]
    final = pool[:final_top_n]
    actually_reranked = False

    if use_rerank and pool:
        try:
            scores = rerank_scores(query, [c.content for c in pool])
            for candidate, score in zip(pool, scores, strict=True):
                candidate.rerank_score = score
            final = sorted(pool, key=lambda c: c.rerank_score or 0.0, reverse=True)[:final_top_n]
            actually_reranked = True
        except Exception as exc:  # noqa: BLE001
            # 重排失败不应该让整个检索挂掉——降级成只用融合结果，用户体验是"稍差"而不是"报错"
            logger.warning("重排失败，降级为融合结果：%s", exc)

    hits = [
        ClauseHit(
            chunk_id=c.chunk_id,
            clause_id=c.clause_id,
            product_id=c.product_id,
            clause_no=c.clause_no,
            title=c.title,
            product_name=c.product_name,
            content=c.content,
            score=c.rerank_score if c.rerank_score is not None else c.rrf_score,
            rrf_score=c.rrf_score,
            dense_rank=c.dense_rank,
            sparse_rank=c.sparse_rank,
            dense_score=c.dense_score,
            sparse_score=c.sparse_score,
            rerank_score=c.rerank_score,
        )
        for c in final
    ]

    return ClauseSearchOut(
        query=query,
        mode=mode,
        used_rerank=actually_reranked,
        hits=hits,
        dense_count=len(dense_candidates),
        sparse_count=len(sparse_candidates),
        merged_count=merged_count,
    )


async def search_clauses_async(query: str, **kwargs) -> ClauseSearchOut:
    """在独立线程里跑同步检索，避免阻塞事件循环。

    search_clauses 内部有 CPU 密集的 embedding 编码和 reranker 精排，
    直接放在 async 路由 / 图节点里会卡死事件循环，导致并发请求互相排队。
    """
    return await asyncio.to_thread(search_clauses, query, **kwargs)
