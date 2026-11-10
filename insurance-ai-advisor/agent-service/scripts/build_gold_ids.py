"""校验评测集并把 (product_code, clause_no) 解析成 gold_clause_id。

评测集里用「产品编码 + 条款编号」做稳定主键（数据库自增 id 会随环境变化），
这里把它解析成真实的 clause_id 写回 gold_set.json，同时校验：

  1. 恰好 50 条，id 连续且不重复；
  2. 每条 gold 都能在数据库里找到对应条款；
  3. 没有两条问题指向同一个条款（避免指标被同一条条款重复拉高）。

用法（在 agent-service 目录下）：
    uv run python scripts/build_gold_ids.py
"""

import asyncio
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import select  # noqa: E402

from app.infrastructure.database import BizSessionLocal, dispose_engines  # noqa: E402
from app.modules.product.models import Clause, Product  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
GOLD_FILE = ROOT / "eval" / "gold_set.json"
EXPECTED = 50


async def run() -> int:
    items = json.loads(GOLD_FILE.read_text(encoding="utf-8"))
    print(f"读取 {GOLD_FILE}，共 {len(items)} 条")

    errors: list[str] = []
    if len(items) != EXPECTED:
        errors.append(f"条数应为 {EXPECTED}，实际 {len(items)}")

    ids = [item["id"] for item in items]
    if ids != list(range(1, EXPECTED + 1)):
        errors.append(f"id 应为 1..{EXPECTED} 连续无重复，实际有重复或缺失")

    async with BizSessionLocal() as session:
        products = (await session.execute(select(Product))).scalars().all()
        by_code = {p.code: p for p in products}

        resolved: list[dict] = []
        for item in items:
            product = by_code.get(item["product_code"])
            if product is None:
                errors.append(f"[id={item['id']}] 产品不存在：{item['product_code']}")
                continue
            clause = await session.scalar(
                select(Clause).where(
                    Clause.product_id == product.id,
                    Clause.clause_no == item["clause_no"],
                )
            )
            if clause is None:
                errors.append(
                    f"[id={item['id']}] 条款不存在：{item['product_code']} {item['clause_no']}"
                )
                continue
            # 只保留评测真正需要的字段：
            #   gold_clause_id 用于命中判定；product_code + clause_no 是稳定业务键，
            #   数据库重建后 id 会变，靠它能重新解析；title / category 供人工复核与分组。
            resolved.append(
                {
                    "id": item["id"],
                    "question": item["question"],
                    "product_code": item["product_code"],
                    "clause_no": item["clause_no"],
                    "title": item["title"],
                    "category": item["category"],
                    "gold_clause_id": clause.id,
                }
            )

    # 重复 gold 检查
    dup = [cid for cid, n in Counter(x["gold_clause_id"] for x in resolved).items() if n > 1]
    if dup:
        errors.append(f"有 {len(dup)} 个条款被多条问题重复引用：{dup}")

    if errors:
        print("\n校验未通过：")
        for err in errors:
            print("  -", err)
        await dispose_engines()
        return 1

    GOLD_FILE.write_text(
        json.dumps(resolved, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"校验通过：{len(resolved)} 条，全部解析到 clause_id，且没有重复的 gold 条款")
    by_cat = Counter(x["category"] for x in resolved)
    print("按险种分布：", dict(by_cat))
    print(f"已写回 {GOLD_FILE}")
    print("字段：id / question / product_code / clause_no / title / category / gold_clause_id")
    await dispose_engines()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run()))
