"""在 50 条标准集上评测条款检索：Top-1 命中率与 MRR。

MRR（Mean Reciprocal Rank）定义：
    对每条问题，找到「第一个命中的正确条款」在结果里的名次 rank，
    该条得分 RR = 1/rank；没命中记 0；所有问题取平均就是 MRR。
    rank=1 得 1 分，rank=2 得 0.5 分，rank=5 得 0.2 分。

Top-1 命中率则只看 rank 是否等于 1。

用法（在 agent-service 目录下）：
    uv run python scripts/evaluate_retrieval.py                     # 默认 hybrid + 重排
    uv run python scripts/evaluate_retrieval.py --mode dense        # 只看语义召回
    uv run python scripts/evaluate_retrieval.py --mode sparse        # 只看 BM25
    uv run python scripts/evaluate_retrieval.py --no-rerank          # 关掉重排
    uv run python scripts/evaluate_retrieval.py --ablation           # 一次跑完 6 种组合
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.logging import setup_logging  # noqa: E402
from app.infrastructure.embedding import get_embedding_model  # noqa: E402
from app.infrastructure.reranker import get_reranker  # noqa: E402
from app.modules.agent.retriever import search_clauses  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
GOLD_FILE = ROOT / "eval" / "gold_set.json"

# 消融矩阵：(检索模式, 是否重排)
ABLATIONS = [
    ("dense", False),
    ("sparse", False),
    ("hybrid", False),
    ("dense", True),
    ("sparse", True),
    ("hybrid", True),
]


def evaluate(items: list[dict], mode: str, use_rerank: bool, top_n: int):
    """跑一轮评测，返回 (top1 命中数, RR 之和, 未命中 id 列表)。"""
    top1 = 0
    rr_sum = 0.0
    misses: list[int] = []

    for item in items:
        result = search_clauses(
            item["question"], mode=mode, final_top_n=top_n, use_rerank=use_rerank
        )
        rank = next(
            (i + 1 for i, hit in enumerate(result.hits) if hit.clause_id == item["gold_clause_id"]),
            None,
        )
        if rank == 1:
            top1 += 1
        if rank:
            rr_sum += 1.0 / rank
        else:
            misses.append(item["id"])

    return top1, rr_sum, misses


def print_row(
    mode: str, rerank: bool, top1: int, rr_sum: float, n: int, misses: list[int], secs: float
):
    print(
        f"{mode:<7} rerank={str(rerank):<5} "
        f"Top1={top1 / n * 100:5.1f}%  MRR={rr_sum / n:.4f}  "
        f"({secs:4.0f}s)  漏检={misses}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="条款检索评测")
    parser.add_argument("--mode", choices=["hybrid", "dense", "sparse"], default="hybrid")
    parser.add_argument("--top-n", type=int, default=5, help="返回条数，也是 MRR 的截断 N")
    parser.add_argument("--no-rerank", action="store_true", help="关闭重排，只看召回质量")
    parser.add_argument("--ablation", action="store_true", help="一次跑完 6 种组合")
    args = parser.parse_args()

    items = json.loads(GOLD_FILE.read_text(encoding="utf-8"))
    n = len(items)

    print("预热模型…")
    get_embedding_model()
    get_reranker()

    if args.ablation:
        print(f"消融矩阵：{n} 条问题，top_n={args.top_n}\n")
        print(f"{'mode':<7} {'rerank':<11} {'Top-1':<9} {'MRR':<8}")
        for mode, rerank in ABLATIONS:
            started = time.perf_counter()
            top1, rr_sum, misses = evaluate(items, mode, rerank, args.top_n)
            print_row(mode, rerank, top1, rr_sum, n, misses, time.perf_counter() - started)
        return 0

    print(f"mode={args.mode} | rerank={not args.no_rerank} | top_n={args.top_n}")
    started = time.perf_counter()
    top1, rr_sum, misses = evaluate(items, args.mode, not args.no_rerank, args.top_n)
    elapsed = time.perf_counter() - started

    print(f"\nTop-1 命中率：{top1}/{n} = {top1 / n * 100:.1f}%")
    print(f"MRR@{args.top_n}：{rr_sum / n:.4f}")
    print(f"未命中（前 {args.top_n} 没有正确条款）：{misses}")
    print(f"总耗时 {elapsed:.1f}s，平均 {elapsed / n * 1000:.0f} ms/条")
    return 0


if __name__ == "__main__":
    setup_logging()
    raise SystemExit(main())
