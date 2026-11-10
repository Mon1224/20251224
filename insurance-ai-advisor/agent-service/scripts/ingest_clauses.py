"""把保险条款切分、向量化、写入 Milvus（第 8 步的离线任务）。

流程：
  1. 从业务库读出条款（连同所属产品，产品名要一起存进 Milvus 便于展示）
  2. 切分：按语义切成约 chunk_size 字的片段，相邻片段留重叠
  3. 切片写入 clause_chunks 表，拿到数据库主键
  4. 用 Qwen3-Embedding-0.6B 把切片编码成 1024 维稠密向量
  5. 写入 Milvus —— 只给稠密向量，稀疏向量由 Milvus 的 BM25 函数自动生成
  6. 把 Milvus 返回的主键回写到 clause_chunks.vector_id，建立双向关联

用法（在 agent-service 目录下）：
    uv run python scripts/ingest_clauses.py             # 增量：只处理没入库的条款
    uv run python scripts/ingest_clauses.py --rebuild   # 全量重建：清空后重来
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import delete, select  # noqa: E402

from app.common.text_splitter import split_text  # noqa: E402
from app.core.config import settings  # noqa: E402
from app.core.logging import setup_logging  # noqa: E402
from app.infrastructure.database import BizSessionLocal, dispose_engines  # noqa: E402
from app.infrastructure.embedding import encode_texts, get_embedding_model  # noqa: E402
from app.infrastructure.milvus import (  # noqa: E402
    collection_stats,
    create_collection,
    get_milvus_client,
    insert_chunks,
)
from app.modules.product.models import Clause, ClauseChunk, Product  # noqa: E402

BATCH_SIZE = 32
logger = logging.getLogger("ingest")


async def run(rebuild: bool) -> None:
    client = get_milvus_client()
    create_collection(client, rebuild=rebuild)

    async with BizSessionLocal() as session:
        if rebuild:
            logger.info("全量重建：清空 clause_chunks 表")
            await session.execute(delete(ClauseChunk))
            # 条款的 chunk_count 也要归零，否则增量模式会以为已经处理过
            rows = (await session.execute(select(Clause))).scalars().all()
            for clause in rows:
                clause.chunk_count = 0
            await session.commit()

        # 1. 读出待处理的条款（连带产品信息）
        result = await session.execute(
            select(Clause, Product)
            .join(Product, Clause.product_id == Product.id)
            .order_by(Clause.id)
        )
        pairs = result.all()

        targets = [(c, p) for c, p in pairs if rebuild or c.chunk_count == 0]
        if not targets:
            logger.info("没有需要处理的条款，全部已入库")
            await report(client)
            return

        logger.info("待处理条款 %s 条，共 %s 条", len(targets), len(pairs))

        # 2. 切分 + 3. 落库，攒成 pending 列表供后面编码
        pending: list[tuple[ClauseChunk, Clause, Product]] = []
        for clause, product in targets:
            pieces = split_text(clause.content, settings.chunk_size, settings.chunk_overlap)
            for index, piece in enumerate(pieces):
                chunk = ClauseChunk(
                    clause_id=clause.id,
                    chunk_index=index,
                    content=piece,
                    token_count=len(piece),
                )
                session.add(chunk)
                pending.append((chunk, clause, product))
            clause.chunk_count = len(pieces)

        await session.commit()  # 提交后 chunk.id 才有值
        logger.info("切分完成，共 %s 个片段", len(pending))

        # 4+5+6. 编码 → 写入 Milvus → 回写主键
        logger.info("加载向量模型（首次约 100 秒，请耐心等待）...")
        get_embedding_model()

        done = 0
        for start in range(0, len(pending), BATCH_SIZE):
            batch = pending[start : start + BATCH_SIZE]
            # 检索文本 = 产品名 + 条款标题 + 正文；dense 与 BM25 都基于它
            search_texts = [
                f"{product.name} {clause.title} {chunk.content}"
                for chunk, clause, product in batch
            ]
            vectors = encode_texts(search_texts)

            rows = [
                {
                    "clause_id": chunk.clause_id,
                    "product_id": product.id,
                    "chunk_index": chunk.chunk_index,
                    "clause_no": clause.clause_no,
                    "title": clause.title,
                    "product_name": product.name,
                    "content": chunk.content,
                    "search_text": search_text,
                    "dense": vector,
                    # sparse 不传：Milvus 会用 search_text 自动生成
                }
                for (chunk, clause, product), search_text, vector in zip(
                    batch, search_texts, vectors, strict=True
                )
            ]

            ids = insert_chunks(client, rows)
            for (chunk, _, _), vector_id in zip(batch, ids, strict=True):
                chunk.vector_id = int(vector_id)

            await session.commit()
            done += len(batch)
            logger.info("进度 %s/%s", done, len(pending))

    client.flush(settings.milvus_collection)
    await report(client)
    await dispose_engines()


async def report(client) -> None:
    stats = collection_stats(client)
    logger.info(
        "Milvus 集合 %s：存在=%s，向量数=%s",
        settings.milvus_collection,
        stats["exists"],
        stats["row_count"],
    )

    async with BizSessionLocal() as session:
        total = len((await session.execute(select(ClauseChunk.id))).scalars().all())
    logger.info("业务库 clause_chunks：%s 行", total)


def main() -> None:
    parser = argparse.ArgumentParser(description="条款向量化入库")
    parser.add_argument("--rebuild", action="store_true", help="清空集合与切片表后全量重建")
    args = parser.parse_args()

    setup_logging()
    asyncio.run(run(rebuild=args.rebuild))


if __name__ == "__main__":
    main()
