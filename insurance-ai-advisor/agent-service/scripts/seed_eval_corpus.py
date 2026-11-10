"""往业务库补充评测用的条款（eval/corpus_extra.json）。

为什么单独放一个脚本：测评集需要一份比演示数据更丰富的条款语料，
但这份语料不该混进 seed_data.py（那是纯演示数据）。这里按
product_code + clause_no 幂等插入，重复执行不会产生重复条款。

用法（在 agent-service 目录下）：
    uv run python scripts/seed_eval_corpus.py
    # 插完再跑一次增量入库，把新条款向量化：
    uv run python scripts/ingest_clauses.py
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import select  # noqa: E402

from app.core.logging import setup_logging  # noqa: E402
from app.infrastructure.database import BizSessionLocal, dispose_engines  # noqa: E402
from app.modules.product.models import Clause, Product  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CORPUS_FILE = ROOT / "eval" / "corpus_extra.json"


async def run() -> None:
    data = json.loads(CORPUS_FILE.read_text(encoding="utf-8"))
    print(f"读取评测语料 {len(data)} 条：{CORPUS_FILE}")

    async with BizSessionLocal() as session:
        products = (await session.execute(select(Product))).scalars().all()
        by_code = {p.code: p for p in products}

        inserted = skipped = 0
        for item in data:
            product = by_code.get(item["product_code"])
            if product is None:
                print(f"  [跳过] 产品不存在：{item['product_code']}")
                continue

            exists = await session.scalar(
                select(Clause.id).where(
                    Clause.product_id == product.id,
                    Clause.clause_no == item["clause_no"],
                )
            )
            if exists:
                skipped += 1
                continue

            session.add(
                Clause(
                    product_id=product.id,
                    clause_no=item["clause_no"],
                    title=item["title"],
                    content=item["content"],
                )
            )
            inserted += 1

        await session.commit()
        print(f"新增条款 {inserted} 条，已存在跳过 {skipped} 条")

    await dispose_engines()


if __name__ == "__main__":
    setup_logging()
    asyncio.run(run())
