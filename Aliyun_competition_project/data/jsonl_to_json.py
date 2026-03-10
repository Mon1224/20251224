import json

IN_JSONL = r"D:\Aliyun_competition_project\data\json_data(1)\pred_dev_lora_rag (1).jsonl"
OUT_JSON = r"D:\Aliyun_competition_project\data\json_data(1)\pred_dev_lora_rag(1).json"

data = []
with open(IN_JSONL, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            data.append(json.loads(line))

with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("saved:", OUT_JSON, "len=", len(data))