import pandas as pd
import random
import os

# =========================
# 参数区
# =========================
RANDOM_SEED = 42
DEV_RATIO = 0.1

REVIEWS_PATH = r"D:\Aliyun_competition_project\data\TRAIN\Train_reviews.csv"
LABELS_PATH = r"D:\Aliyun_competition_project\data\TRAIN\Train_labels.csv"

OUTPUT_DIR = "dataset_split(1)"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# 1. 读取 CSV
# =========================
print("读取数据中...")
reviews_df = pd.read_csv(REVIEWS_PATH)
labels_df = pd.read_csv(LABELS_PATH)

# 确保有 ID 字段
assert "id" in reviews_df.columns, "Train_reviews.csv 缺少 id 列"
assert "id" in labels_df.columns, "Train_labels.csv 缺少 id 列"

# =========================
# 2. 获取唯一 ID 并打乱
# =========================
all_ids = reviews_df["id"].unique().tolist()

print(f"总评论数: {len(all_ids)}")

random.seed(RANDOM_SEED)
random.shuffle(all_ids)

# =========================
# 3. 划分 Train / Dev
# =========================
dev_size = int(len(all_ids) * DEV_RATIO)

dev_ids = all_ids[:dev_size]
train_ids = all_ids[dev_size:]

print(f"Train 评论数: {len(train_ids)}")
print(f"Dev 评论数: {len(dev_ids)}")

# =========================
# 4. 保存 dev_ids 以保证复现
# =========================
with open(os.path.join(OUTPUT_DIR, "dev_ids.txt"), "w", encoding="utf-8") as f:
    for _id in dev_ids:
        f.write(str(_id) + "\n")

# =========================
# 5. 根据 ID 筛选 reviews
# =========================
train_reviews = reviews_df[reviews_df["id"].isin(train_ids)]
dev_reviews = reviews_df[reviews_df["id"].isin(dev_ids)]

# =========================
# 6. 根据 ID 筛选 labels
# =========================
train_labels = labels_df[labels_df["id"].isin(train_ids)]
dev_labels = labels_df[labels_df["id"].isin(dev_ids)]

# =========================
# 7. 保存 CSV
# =========================
train_reviews.to_csv(os.path.join(OUTPUT_DIR, "train_reviews.csv"), index=False, encoding="utf-8-sig")
dev_reviews.to_csv(os.path.join(OUTPUT_DIR, "dev_reviews.csv"), index=False, encoding="utf-8-sig")

train_labels.to_csv(os.path.join(OUTPUT_DIR, "train_labels.csv"), index=False, encoding="utf-8-sig")
dev_labels.to_csv(os.path.join(OUTPUT_DIR, "dev_labels.csv"), index=False, encoding="utf-8-sig")

print("\n数据集划分完成！输出目录：dataset_split/")
