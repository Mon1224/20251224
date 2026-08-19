import json
import pandas as pd
from collections import defaultdict

# ====== 路径配置（建议用绝对路径）======
PRED_JSON = r"D:\Aliyun_competition_project\data\json_data(1)\pred_dev_lora_rag(1).json"   # 你的推理输出
DEV_LABELS_CSV = r"D:\Aliyun_competition_project\data\dataset_split(1)\dev_labels.csv"       # 你的真值标签（CSV）
# 如果你是本地跑，改成你的实际路径即可

# ====== 字段名配置（如与你CSV不一致就改这里）======)
ID_COL = "id"
ASPECT_COL = "AspectTerms"
OPINION_COL = "OpinionTerms"
POLARITY_COL = "Polarities"
CATEGORY_COL = "Categories"

def norm(s):
    """严格匹配时通常不做太多归一化，这里只做最基础的strip。"""
    if s is None:
        return ""
    return str(s).strip()

def quad_to_tuple(a, o, c, p):
    """四字段全对才算对：AspectTerm、OpinionTerm、Category、Polarity"""
    return (norm(a), norm(o), norm(c), norm(p))

def load_gold_from_csv(labels_csv_path):
    """读 dev_labels.csv，按ID聚合为 set(quadruple)"""
    df = pd.read_csv(labels_csv_path)
    gold = defaultdict(set)
    for _, r in df.iterrows():
        rid = int(r[ID_COL])
        gold[rid].add(quad_to_tuple(r[ASPECT_COL], r[OPINION_COL], r[CATEGORY_COL], r[POLARITY_COL]))
    return gold

def load_pred_from_json(pred_json_path):
    """读 pred_dev.json，按ID聚合为 set(quadruple)"""
    with open(pred_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    pred = defaultdict(set)
    for item in data:
        rid = int(item["id"])
        quads = item.get("prediction", []) or []
        for q in quads:
            pred[rid].add(
                quad_to_tuple(
                    q.get("aspect", ""),
                    q.get("opinion", ""),
                    q.get("category", ""),
                    q.get("polarity", "")
                )
            )
    return pred

def compute_f1(gold, pred):
    """微平均：全体四元组层面统计"""
    correct = 0
    total_pred = 0
    total_gold = 0

    all_ids = set(gold.keys()) | set(pred.keys())
    per_id_stats = {}

    for rid in all_ids:
        g = gold.get(rid, set())
        p = pred.get(rid, set())

        c = len(g & p)
        total_pred += len(p)
        total_gold += len(g)
        correct += c

        per_id_stats[rid] = {
            "gold": len(g),
            "pred": len(p),
            "correct": c
        }

    precision = correct / total_pred if total_pred > 0 else 0.0
    recall = correct / total_gold if total_gold > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return precision, recall, f1, correct, total_pred, total_gold, per_id_stats

def main():
    gold = load_gold_from_csv(DEV_LABELS_CSV)
    pred = load_pred_from_json(PRED_JSON)

    precision, recall, f1, correct, total_pred, total_gold, per_id_stats = compute_f1(gold, pred)

    print("==== Strict Quadruple F1 (Exact Match on 4 fields) ====")
    print(f"Correct: {correct}")
    print(f"Pred total: {total_pred}")
    print(f"Gold total: {total_gold}")
    print(f"Precision: {precision:.6f}")
    print(f"Recall:    {recall:.6f}")
    print(f"F1:        {f1:.6f}")

    # 可选：输出最容易出错的样本（预测多但命中少）
    hard = sorted(per_id_stats.items(), key=lambda x: (-(x[1]["pred"] - x[1]["correct"]), x[1]["correct"]))[:20]
    print("\n---- Hard cases (top 20 by (pred-correct)) ----")
    for rid, st in hard:
        if st["pred"] == 0 and st["gold"] == 0:
            continue
        print(f"ID={rid} gold={st['gold']} pred={st['pred']} correct={st['correct']}")

if __name__ == "__main__":
    main()