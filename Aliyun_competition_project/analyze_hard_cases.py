import os
import json
import pandas as pd
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set, Any, Optional

# =========================
# Paths (你的路径)
# =========================
DEV_REVIEWS_CSV = r"D:\Aliyun_competition_project\data\dataset_split(1)\dev_reviews.csv"
DEV_LABELS_CSV  = r"D:\Aliyun_competition_project\data\dataset_split(1)\dev_labels.csv"

# 这两个二选一，脚本会自动挑存在的那个
PRED_JSON  = r"D:\Aliyun_competition_project\data\json_data(1)\pred_dev_lora.json"
PRED_JSONL = r"D:\Aliyun_competition_project\data\json_data(1)\pred_dev_lora.jsonl"

OUT_DIR = r"D:\Aliyun_competition_project\data\analysis_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# =========================
# Column config
# =========================
ID_COL = "id"
REVIEW_COL = "Reviews"

ASPECT_COL = "AspectTerms"
OPINION_COL = "OpinionTerms"
CATEGORY_COL = "Categories"
POLARITY_COL = "Polarities"

# 你想分析的 hard IDs：先用你给的那批
HARD_IDS = [341, 505, 3180, 1185, 2127, 2248, 364, 2429]

# 可选：如果你想分析“评测脚本输出的 top20 hard”，把它们直接粘贴到这里替换 HARD_IDS 即可


# =========================
# Helpers
# =========================
def norm(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip()

def quad_tuple(a, o, c, p) -> Tuple[str, str, str, str]:
    return (norm(a), norm(o), norm(c), norm(p))

def read_reviews(path: str) -> Dict[int, str]:
    df = pd.read_csv(path)
    m = {}
    for _, r in df.iterrows():
        rid = int(r[ID_COL])
        m[rid] = str(r[REVIEW_COL])
    return m

def read_gold_labels(path: str) -> Dict[int, Set[Tuple[str, str, str, str]]]:
    df = pd.read_csv(path)
    gold = defaultdict(set)
    for _, r in df.iterrows():
        rid = int(r[ID_COL])
        gold[rid].add(quad_tuple(r[ASPECT_COL], r[OPINION_COL], r[CATEGORY_COL], r[POLARITY_COL]))
    return gold

def load_pred(path_json: str, path_jsonl: str) -> Dict[int, Set[Tuple[str, str, str, str]]]:
    pred = defaultdict(set)

    if os.path.exists(path_json):
        with open(path_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            rid = int(item["id"])
            quads = item.get("prediction", []) or []
            for q in quads:
                pred[rid].add(quad_tuple(q.get("aspect",""), q.get("opinion",""), q.get("category",""), q.get("polarity","")))
        return pred

    if os.path.exists(path_jsonl):
        with open(path_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                rid = int(item["id"])
                quads = item.get("prediction", []) or []
                for q in quads:
                    pred[rid].add(quad_tuple(q.get("aspect",""), q.get("opinion",""), q.get("category",""), q.get("polarity","")))
        return pred

    raise FileNotFoundError(f"Neither pred json nor jsonl exists:\n{path_json}\n{path_jsonl}")

def best_match_by_opinion(pred_q: Tuple[str,str,str,str], gold_set: Set[Tuple[str,str,str,str]]) -> Optional[Tuple[str,str,str,str]]:
    """先按 opinion 精确匹配找候选，再选字段最接近的那条。"""
    pa, po, pc, pp = pred_q
    cands = [g for g in gold_set if g[1] == po]
    if not cands:
        return None
    # 选差异最少的
    def dist(g):
        ga, go, gc, gp = g
        d = 0
        d += (ga != pa)
        d += (gc != pc)
        d += (gp != pp)
        return d
    return sorted(cands, key=dist)[0]

def classify_pred_only(pred_q: Tuple[str,str,str,str], gold_set: Set[Tuple[str,str,str,str]]) -> str:
    """
    给 pred-only 的四元组打错因标签（粗粒度，够做PPT/指导RAG）。
    """
    pa, po, pc, pp = pred_q

    # 1) opinion 完全没在 gold 里：多抽/边界/同义
    gold_opinions = {g[1] for g in gold_set}
    if po not in gold_opinions:
        return "opinion_not_in_gold (spurious_or_boundary_or_synonym)"

    # 2) opinion 在 gold 里，找同 opinion 的最佳 gold 来对比字段
    g = best_match_by_opinion(pred_q, gold_set)
    if g is None:
        return "opinion_match_failed (unexpected)"

    ga, go, gc, gp = g
    wrongs = []
    if pa != ga:
        wrongs.append("aspect_wrong")
    if pc != gc:
        wrongs.append("category_wrong")
    if pp != gp:
        wrongs.append("polarity_wrong")

    if not wrongs:
        # 理论上不会 pred-only 出现完全一致
        return "should_have_matched (dup_or_bug)"

    # 多字段错：通常是配对问题或类别/极性整体偏移
    if len(wrongs) >= 2:
        return "pairing_mismatch_or_multi_field_wrong: " + "+".join(wrongs)

    return wrongs[0]

def classify_gold_only(gold_q: Tuple[str,str,str,str], pred_set: Set[Tuple[str,str,str,str]]) -> str:
    """
    给 gold-only（漏掉）的四元组打粗错因标签。
    """
    ga, go, gc, gp = gold_q
    pred_opinions = {p[1] for p in pred_set}

    if go not in pred_opinions:
        return "missed_opinion (model_didnt_extract_or_boundary)"

    # opinion 出现了但没命中：说明字段错/配对错
    cands = [p for p in pred_set if p[1] == go]
    if not cands:
        return "opinion_present_but_no_candidate (unexpected)"

    # 找最接近的一条看看错在哪
    def dist(p):
        pa, po, pc, pp = p
        d = 0
        d += (pa != ga)
        d += (pc != gc)
        d += (pp != gp)
        return d
    p = sorted(cands, key=dist)[0]
    pa, po, pc, pp = p

    wrongs = []
    if pa != ga:
        wrongs.append("aspect_wrong")
    if pc != gc:
        wrongs.append("category_wrong")
    if pp != gp:
        wrongs.append("polarity_wrong")

    if len(wrongs) >= 2:
        return "pairing_mismatch_or_multi_field_wrong: " + "+".join(wrongs)
    if len(wrongs) == 1:
        return wrongs[0]
    return "should_have_matched (dup_or_bug)"

def fmt_quads(quads: Set[Tuple[str,str,str,str]]) -> str:
    if not quads:
        return "  (none)\n"
    lines = []
    for a,o,c,p in sorted(quads):
        lines.append(f'  - (A="{a}", O="{o}", C="{c}", P="{p}")')
    return "\n".join(lines) + "\n"


# =========================
# Main
# =========================
def main():
    reviews = read_reviews(DEV_REVIEWS_CSV)
    gold = read_gold_labels(DEV_LABELS_CSV)
    pred = load_pred(PRED_JSON, PRED_JSONL)

    # 全局错误统计（在 hard IDs 上也做一份；你也可以改成全 dev）
    pred_only_counter = Counter()
    gold_only_counter = Counter()

    report_lines: List[str] = []
    report_lines.append("==== Hard Case Analysis Report ====\n")

    for rid in HARD_IDS:
        rtext = reviews.get(rid, "(review not found)")
        gset = gold.get(rid, set())
        pset = pred.get(rid, set())

        hit = gset & pset
        pred_only = pset - gset
        gold_only = gset - pset

        report_lines.append(f"\n==============================")
        report_lines.append(f"ID = {rid}")
        report_lines.append(f"REVIEW: {rtext}\n")
        report_lines.append(f"[Gold] {len(gset)} quads:\n{fmt_quads(gset)}")
        report_lines.append(f"[Pred] {len(pset)} quads:\n{fmt_quads(pset)}")
        report_lines.append(f"[Hit] {len(hit)} quads:\n{fmt_quads(hit)}")

        # pred-only 分类
        if pred_only:
            report_lines.append(f"[Pred-only] {len(pred_only)} quads (extra/wrong):")
            for q in sorted(pred_only):
                tag = classify_pred_only(q, gset)
                pred_only_counter[tag] += 1
                report_lines.append(f"  * {q}  =>  {tag}")
        else:
            report_lines.append("[Pred-only] 0")

        # gold-only 分类
        if gold_only:
            report_lines.append(f"[Gold-only] {len(gold_only)} quads (missed):")
            for q in sorted(gold_only):
                tag = classify_gold_only(q, pset)
                gold_only_counter[tag] += 1
                report_lines.append(f"  * {q}  =>  {tag}")
        else:
            report_lines.append("[Gold-only] 0")

    # 输出报告 txt
    report_path = os.path.join(OUT_DIR, "hardcase_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    # 输出错误类型汇总 csv（便于 WPS 做图）
    rows = []
    for k,v in pred_only_counter.items():
        rows.append({"side": "pred_only", "error_type": k, "count": v})
    for k,v in gold_only_counter.items():
        rows.append({"side": "gold_only", "error_type": k, "count": v})
    df = pd.DataFrame(rows).sort_values(["side", "count"], ascending=[True, False])
    csv_path = os.path.join(OUT_DIR, "error_breakdown.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print("Done.")
    print(f"Saved: {report_path}")
    print(f"Saved: {csv_path}")
    print("\nTop error types (pred_only):")
    for k,v in pred_only_counter.most_common(10):
        print(f"  {k}: {v}")
    print("\nTop error types (gold_only):")
    for k,v in gold_only_counter.most_common(10):
        print(f"  {k}: {v}")

if __name__ == "__main__":
    main()