import os
import json
import re
import pickle
import argparse
from collections import defaultdict
from typing import List, Dict, Any, Tuple

import pandas as pd

# -------------------------
# Tokenizers
# -------------------------
def normalize_text(s: str) -> str:
    s = "" if s is None else str(s)
    # 轻度清洗：去掉多余空白；保留中文标点（BM25 可用）
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tokenize_char(text: str) -> List[str]:
    text = normalize_text(text)
    # 字符级：过滤空格
    return [ch for ch in text if ch.strip()]

def tokenize_jieba(text: str) -> List[str]:
    text = normalize_text(text)
    import jieba  # lazy import
    # 精确模式，过滤空白 token
    return [t for t in jieba.lcut(text, cut_all=False) if t.strip()]

# -------------------------
# Build train_meta.json
# -------------------------
def build_train_meta(
    reviews_csv: str,
    labels_csv: str,
    id_col_reviews: str = "id",
    review_col: str = "Reviews",
    id_col_labels: str = "id",
    aspect_col: str = "AspectTerms",
    opinion_col: str = "OpinionTerms",
    category_col: str = "Categories",
    polarity_col: str = "Polarities",
) -> List[Dict[str, Any]]:
    df_r = pd.read_csv(reviews_csv)
    df_l = pd.read_csv(labels_csv)

    # reviews map
    rid_to_review: Dict[int, str] = {}
    for _, r in df_r.iterrows():
        rid = int(r[id_col_reviews])
        rid_to_review[rid] = normalize_text(r[review_col])

    # labels grouped
    rid_to_labels: Dict[int, List[Dict[str, str]]] = defaultdict(list)
    for _, r in df_l.iterrows():
        rid = int(r[id_col_labels])
        a = "" if pd.isna(r[aspect_col]) else str(r[aspect_col]).strip()
        o = "" if pd.isna(r[opinion_col]) else str(r[opinion_col]).strip()
        c = "" if pd.isna(r[category_col]) else str(r[category_col]).strip()
        p = "" if pd.isna(r[polarity_col]) else str(r[polarity_col]).strip()
        # 统一：空值当 "_"（你数据本来也是这样）
        a = a if a else "_"
        o = o if o else "_"
        c = c if c else "_"
        p = p if p else "_"
        rid_to_labels[rid].append(
            {"aspect": a, "opinion": o, "category": c, "polarity": p}
        )

    # assemble
    meta: List[Dict[str, Any]] = []
    missing_review = 0
    missing_label = 0

    all_ids = sorted(set(rid_to_review.keys()) | set(rid_to_labels.keys()))
    for rid in all_ids:
        review = rid_to_review.get(rid)
        labels = rid_to_labels.get(rid, [])
        if review is None:
            missing_review += 1
            continue
        if not labels:
            # 训练集一般不会没有 label；但稳健起见保留空
            missing_label += 1
        meta.append({"id": rid, "review": review, "labels": labels})

    print(f"[Meta] total ids: {len(all_ids)}")
    print(f"[Meta] kept items: {len(meta)}")
    print(f"[Meta] missing reviews dropped: {missing_review}")
    print(f"[Meta] missing labels kept as empty: {missing_label}")
    return meta

# -------------------------
# Build BM25 index
# -------------------------
def build_bm25(meta: List[Dict[str, Any]], tokenizer: str):
    try:
        from rank_bm25 import BM25Okapi
    except Exception as e:
        raise RuntimeError(
            "Missing dependency: rank-bm25. Install with `pip install rank-bm25`."
        ) from e

    if tokenizer == "jieba":
        tok_fn = tokenize_jieba
    elif tokenizer == "char":
        tok_fn = tokenize_char
    else:
        raise ValueError("tokenizer must be 'jieba' or 'char'")

    corpus_tokens: List[List[str]] = []
    for item in meta:
        corpus_tokens.append(tok_fn(item["review"]))

    bm25 = BM25Okapi(corpus_tokens)
    return bm25, corpus_tokens

# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_reviews", required=True, help="path to train_reviews.csv")
    ap.add_argument("--train_labels", required=True, help="path to train_labels.csv")
    ap.add_argument("--out_dir", required=True, help="output dir")
    ap.add_argument("--tokenizer", default="jieba", choices=["jieba", "char"])

    # columns (keep defaults matching your files)
    ap.add_argument("--id_col_reviews", default="id")
    ap.add_argument("--review_col", default="Reviews")
    ap.add_argument("--id_col_labels", default="id")
    ap.add_argument("--aspect_col", default="AspectTerms")
    ap.add_argument("--opinion_col", default="OpinionTerms")
    ap.add_argument("--category_col", default="Categories")
    ap.add_argument("--polarity_col", default="Polarities")

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    train_meta_path = os.path.join(args.out_dir, "train_meta.json")
    bm25_pkl_path = os.path.join(args.out_dir, "bm25.pkl")
    tokens_dump_path = os.path.join(args.out_dir, "bm25_corpus_tokens.jsonl")

    # 1) meta
    meta = build_train_meta(
        reviews_csv=args.train_reviews,
        labels_csv=args.train_labels,
        id_col_reviews=args.id_col_reviews,
        review_col=args.review_col,
        id_col_labels=args.id_col_labels,
        aspect_col=args.aspect_col,
        opinion_col=args.opinion_col,
        category_col=args.category_col,
        polarity_col=args.polarity_col,
    )

    with open(train_meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[OK] saved: {train_meta_path}")

    # 2) bm25 index
    bm25, corpus_tokens = build_bm25(meta, tokenizer=args.tokenizer)

    with open(bm25_pkl_path, "wb") as f:
        pickle.dump(
            {
                "tokenizer": args.tokenizer,
                "bm25": bm25,
                "id_list": [int(x["id"]) for x in meta],  # 对齐 corpus 顺序
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    print(f"[OK] saved: {bm25_pkl_path}")

    # 3) optional tokens dump for debugging
    with open(tokens_dump_path, "w", encoding="utf-8") as f:
        for item, toks in zip(meta, corpus_tokens):
            f.write(
                json.dumps(
                    {"id": int(item["id"]), "tokens": toks},
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"[OK] saved: {tokens_dump_path}")

    print("\nDone.")
    print("Outputs:")
    print(f"  - train_meta.json: {train_meta_path}")
    print(f"  - bm25.pkl:        {bm25_pkl_path}")

if __name__ == "__main__":
    main()