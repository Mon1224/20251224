import json
import pandas as pd
from collections import defaultdict, Counter

# ====== 路径（按你的实际路径改）======
PRED_JSON = r"D:\Aliyun_competition_project\data\json_data(1)\pred_dev_lora_rag(1).json"
DEV_REVIEWS_CSV = r"D:\Aliyun_competition_project\data\dataset_split(1)\dev_reviews.csv"
DEV_LABELS_CSV = r"D:\Aliyun_competition_project\data\dataset_split(1)\dev_labels.csv"

# ====== 列名（你说 reviews 列名是 Reviews）======
ID_COL = "id"
REVIEW_COL = "Reviews"

ASPECT_COL = "AspectTerms"
OPINION_COL = "OpinionTerms"
POLARITY_COL = "Polarities"
CATEGORY_COL = "Categories"


def norm(s):
    if s is None:
        return ""
    return str(s).strip()


def load_reviews_map(path):
    df = pd.read_csv(path)
    assert ID_COL in df.columns, f"{path} 缺少列 {ID_COL}"
    assert REVIEW_COL in df.columns, f"{path} 缺少列 {REVIEW_COL}"
    mp = {}
    for _, r in df.iterrows():
        mp[int(r[ID_COL])] = str(r[REVIEW_COL])
    return mp


def load_gold_sets(path):
    df = pd.read_csv(path)
    assert ID_COL in df.columns, f"{path} 缺少列 {ID_COL}"
    gold = defaultdict(list)  # keep list (not set) for counting
    cat_set = set()
    pol_set = set()
    total = 0
    for _, r in df.iterrows():
        rid = int(r[ID_COL])
        a = norm(r[ASPECT_COL])
        o = norm(r[OPINION_COL])
        c = norm(r[CATEGORY_COL])
        p = norm(r[POLARITY_COL])
        gold[rid].append((a, o, c, p))
        cat_set.add(c)
        pol_set.add(p)
        total += 1
    return gold, cat_set, pol_set, total


def load_pred_sets(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    pred = defaultdict(list)  # keep list for counting + duplicates
    raw_wrap_count = 0
    raw_think_count = 0

    for item in data:
        rid = int(item["id"])
        raw = str(item.get("prediction_raw", ""))
        if '"quadruples"' in raw:
            raw_wrap_count += 1
        if "<think>" in raw or "</think>" in raw:
            raw_think_count += 1

        quads = item.get("prediction", []) or []
        for q in quads:
            a = norm(q.get("aspect", ""))
            o = norm(q.get("opinion", ""))
            c = norm(q.get("category", ""))
            p = norm(q.get("polarity", ""))
            pred[rid].append((a, o, c, p))

    return pred, raw_wrap_count, raw_think_count, len(data)


def is_substring(span, text):
    """span 是否是 text 的连续子串。允许 span 为 '_' 或空时直接视为 True。"""
    span = norm(span)
    if span == "" or span == "_":
        return True
    if text is None:
        return False
    return span in text


def main():
    reviews = load_reviews_map(DEV_REVIEWS_CSV)
    gold, gold_cat_set, gold_pol_set, gold_total = load_gold_sets(DEV_LABELS_CSV)
    pred, raw_wrap_count, raw_think_count, pred_item_count = load_pred_sets(PRED_JSON)

    all_ids = set(reviews.keys()) | set(gold.keys()) | set(pred.keys())

    # ====== 统计：数量层面（漏抽/多抽）======
    pred_total = sum(len(pred.get(rid, [])) for rid in all_ids)
    gold_total2 = sum(len(gold.get(rid, [])) for rid in all_ids)

    avg_pred_per_review = pred_total / max(1, len(all_ids))
    avg_gold_per_review = gold_total2 / max(1, len(all_ids))

    # ====== 统计：枚举对齐 ======
    pred_cat_counter = Counter()
    pred_pol_counter = Counter()

    pred_cat_oov = 0
    pred_pol_oov = 0

    for rid in all_ids:
        for (a, o, c, p) in pred.get(rid, []):
            pred_cat_counter[c] += 1
            pred_pol_counter[p] += 1
            if c not in gold_cat_set:
                pred_cat_oov += 1
            if p not in gold_pol_set:
                pred_pol_oov += 1

    # ====== 统计：子串约束（是否“复制原文”）======
    aspect_not_in_text = 0
    opinion_not_in_text = 0
    total_pred_quads = 0

    # 同时统计最常见的“非子串”内容，方便你看问题集中在哪些词
    aspect_not_in_text_counter = Counter()
    opinion_not_in_text_counter = Counter()

    for rid in all_ids:
        text = reviews.get(rid, "")
        for (a, o, c, p) in pred.get(rid, []):
            total_pred_quads += 1
            if not is_substring(a, text):
                aspect_not_in_text += 1
                aspect_not_in_text_counter[a] += 1
            if not is_substring(o, text):
                opinion_not_in_text += 1
                opinion_not_in_text_counter[o] += 1

    # ====== 统计：每条样本的“覆盖情况” ======
    # 命中率上限估计：如果你预测数远小于真值数，即使全对也上不了高 recall
    # 这里只做一个直观统计：多少条评论 pred=0
    pred_zero = sum(1 for rid in all_ids if len(pred.get(rid, [])) == 0)
    gold_zero = sum(1 for rid in all_ids if len(gold.get(rid, [])) == 0)

    # ====== 输出诊断报告 ======
    print("==== Diagnose Base Predictions ====")
    print(f"Reviews count (union ids): {len(all_ids)}")
    print(f"Gold quadruples total:     {gold_total2}")
    print(f"Pred quadruples total:     {pred_total}")
    print(f"Avg gold per review:       {avg_gold_per_review:.3f}")
    print(f"Avg pred per review:       {avg_pred_per_review:.3f}")
    print(f"Reviews with pred=0:        {pred_zero} ({pred_zero/len(all_ids):.1%})")
    print(f"Reviews with gold=0:        {gold_zero} ({gold_zero/len(all_ids):.1%})")

    print("\n---- Output format signals (from prediction_raw) ----")
    print(f'Contains "<think>":        {raw_think_count}/{pred_item_count} ({raw_think_count/max(1,pred_item_count):.1%})')

    print("\n---- Category / Polarity enum alignment ----")
    print(f"Gold Category unique:      {len(gold_cat_set)}")
    print(f"Gold Polarity unique:      {len(gold_pol_set)}")
    print(f"Pred Category OOV count:   {pred_cat_oov}/{max(1,total_pred_quads)} ({pred_cat_oov/max(1,total_pred_quads):.1%})")
    print(f"Pred Polarity OOV count:   {pred_pol_oov}/{max(1,total_pred_quads)} ({pred_pol_oov/max(1,total_pred_quads):.1%})")

    print("\nTop-15 predicted Categories:")
    for k, v in pred_cat_counter.most_common(15):
        mark = "" if k in gold_cat_set else "  <-- OOV"
        print(f"  {k!r}: {v}{mark}")

    print("\nTop-10 predicted Polarities:")
    for k, v in pred_pol_counter.most_common(10):
        mark = "" if k in gold_pol_set else "  <-- OOV"
        print(f"  {k!r}: {v}{mark}")

    print("\n---- Substring (copy-from-review) checks ----")
    print(f"Aspect not substring:      {aspect_not_in_text}/{max(1,total_pred_quads)} ({aspect_not_in_text/max(1,total_pred_quads):.1%})")
    print(f"Opinion not substring:     {opinion_not_in_text}/{max(1,total_pred_quads)} ({opinion_not_in_text/max(1,total_pred_quads):.1%})")

    print("\nTop-10 aspects NOT in review text:")
    for k, v in aspect_not_in_text_counter.most_common(10):
        print(f"  {k!r}: {v}")

    print("\nTop-10 opinions NOT in review text:")
    for k, v in opinion_not_in_text_counter.most_common(10):
        print(f"  {k!r}: {v}")

    print("\n==== Suggested next actions based on results ====")
    print("1) 如果 Pred Category OOV 比例高：把 prompt 的 category 枚举换成 gold 的全集（严格一致）。")
    print("2) 如果 pred 总数远小于 gold 总数：推理改为 do_sample=False，并明确“尽可能抽全所有四元组”，同时增大 max_new_tokens。")
    print("3) 如果 aspect/opinion 非子串比例高：prompt 强制“必须复制原文连续片段”；第二阶段 Base 用校验/回填进一步提 precision。")


if __name__ == "__main__":
    main()