import pandas as pd
import json
import os

def convert_to_json(review_path, label_path, output_path):
    print(f"处理: {review_path}")

    reviews_df = pd.read_csv(review_path)
    labels_df = pd.read_csv(label_path)

    # 按 ID 分组 labels
    label_groups = labels_df.groupby("id")

    result = []

    for _, row in reviews_df.iterrows():
        rid = row["id"]
        review_text = row["Reviews"]

        labels_list = []

        if rid in label_groups.groups:
            group = label_groups.get_group(rid)

            for _, l in group.iterrows():
                labels_list.append({
                    "aspect": str(l["AspectTerms"]),
                    "aspect_start": -1 if pd.isna(l["A_start"]) or str(l["A_start"]).strip()=="" else int(l["A_start"]),
                    "aspect_end": -1 if pd.isna(l["A_end"]) or str(l["A_end"]).strip()=="" else int(l["A_end"]),
                    "opinion": str(l["OpinionTerms"]),
                    "opinion_start": -1 if pd.isna(l["O_start"]) or str(l["O_start"]).strip()=="" else int(l["O_start"]),
                    "opinion_end": -1 if pd.isna(l["O_end"]) or str(l["O_end"]).strip()=="" else int(l["O_end"]),
                    "polarity": str(l["Polarities"]),
                    "category": str(l["Categories"])
                })

        result.append({
            "id": int(rid),
            "review": review_text,
            "labels": labels_list
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"已生成: {output_path}")


# ==========================
# 主程序
# ==========================
os.makedirs("json_data(1)", exist_ok=True)

convert_to_json(
    r"D:\Aliyun_competition_project\data\dataset_split(1)\train_reviews.csv",
    r"D:\Aliyun_competition_project\data\dataset_split(1)\train_labels.csv",
    r"D:\Aliyun_competition_project\data\json_data(1)\train.json"
)

convert_to_json(
    r"D:\Aliyun_competition_project\data\dataset_split(1)\dev_reviews.csv",
    r"D:\Aliyun_competition_project\data\dataset_split(1)\dev_labels.csv",
    r"D:\Aliyun_competition_project\data\json_data(1)\dev.json"
)
