import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import re
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen3-8B"
DEV_JSON = r"D:\Aliyun_competition_project\data\json_data\dev.json"
OUT_JSON = r"D:\Aliyun_competition_project\data\json_data\pred_dev.json"

CATEGORY_LIST = ["包装","成分","尺寸","服务","功效","价格","气味","使用体验","物流","新鲜度","真伪","整体","其他"]
POLARITY_LIST = ["正面", "中性", "负面"]

CATEGORY_SET = set(CATEGORY_LIST)
POLARITY_SET = set(POLARITY_LIST)


def build_messages(review: str, few_shots: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, str]]:
    system = (
        "你是一个专业的中文电商化妆品评论观点四元组抽取助手。\n"
        "任务：给定一条化妆品电商评论文本，抽取其中所有的观点四元组（AspectTerm, OpinionTerm, Category, Polarity），即 ACOS 四元组。\n"
        "严格遵守以下规则：\n"
        "1.输出必须严格是一个 JSON 数组，数组元素为："
        '[{"aspect":"...","opinion":"...","category":"...","polarity":"..."}]\n'
        "2.如果没有观点，输出 []。\n"
        "3.严禁输出 <think> 或任何解释文字，若必须思考请在内部完成，不要输出任何思考内容。\n"
        "4. aspect 必须是评论原文中出现的连续片段，必须严格与原文一致，不得修改，不添加空格；若评论里没有明确属性词，aspect 必须输出 \"_\"。\n"
        "5.绝对禁止把 category 名称（如“使用体验/功效/整体/价格/服务/物流/包装/气味/真伪/成分/尺寸/新鲜度/其他”）当作 aspect。\n"
        "6.opinion 必须是评论原文中出现的连续片段，保持原字符，不得修改，严格与原文一致；无观点则不输出该条四元组。\n"
        "7.category取值仅限：包装、成分、尺寸、服务、功效、价格、气味、使用体验、物流、新鲜度、真伪、整体、其他。\n"
        "8.polarity取值仅限：正面、中性、负面。\n"
        "9.尽可能完整抽取所有四元组；同一评论可能有多个四元组。\n"
        "10.去重：同一(aspect, opinion, category, polarity)组合只保留一个。\n"
        "11.排序：按opinion在原文首次出现位置升序；若相同，按aspect排序（\"_\"视为最大）。\n"
        "12.不得出现与原文无关或臆造的词；不得输出解释文字；不得添加多余字段。\n"
        "13.只输出JSON，不输出其他任何文本。\n"
    )

    user_parts = []
    if few_shots:
        user_parts.append("示例：")
        for ex in few_shots:
            # few-shot 里只放四字段，别放 start/end，避免模型学到多余字段
            user_parts.append(
                f"评论：{ex['review']}\n输出：{json.dumps(ex['labels'], ensure_ascii=False)}\n"
            )
    user_parts.append("现在抽取下列评论，仅输出 JSON 数组：")
    user_parts.append(f"评论：{review}")

    return [{"role": "system", "content": system}, {"role": "user", "content": "\n".join(user_parts)}]


def extract_quads_robust(text: str) -> List[Dict[str, Any]]:
    """
    1) 直接 json.loads
    2) 抓完整的 [...] 再 loads
    3) 抓完整的 {...} 再尝试 dict["quadruples"]
    4) 截断容错：捡回多个完整的 {...}（即使数组没闭合）
    """
    text = strip_think(text)

    # 1) direct loads
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            if "quadruples" in obj and isinstance(obj["quadruples"], list):
                return obj["quadruples"]
            for k in ("prediction", "predictions", "labels"):
                if k in obj and isinstance(obj[k], list):
                    return obj[k]
    except Exception:
        pass

    # 2) complete array
    m_arr = re.search(r"\[[\s\S]*\]", text)
    if m_arr:
        snippet = m_arr.group(0)
        try:
            obj = json.loads(snippet)
            if isinstance(obj, list):
                return obj
        except Exception:
            pass

    # 3) complete object
    m_obj = re.search(r"\{[\s\S]*\}", text)
    if m_obj:
        snippet = m_obj.group(0)
        try:
            obj = json.loads(snippet)
            if isinstance(obj, dict) and "quadruples" in obj and isinstance(obj["quadruples"], list):
                return obj["quadruples"]
        except Exception:
            pass

    # 4) salvage complete tiny dict objects
    salvaged = []
    for m in re.finditer(r"\{[^{}]*\}", text):
        snippet = m.group(0)
        try:
            o = json.loads(snippet)
            if isinstance(o, dict) and {"aspect", "opinion", "category", "polarity"} <= set(o.keys()):
                salvaged.append(o)
        except Exception:
            continue

    return salvaged


def norm(s: Any) -> str:
    return str(s).strip() if s is not None else ""


def postprocess_quads(review: str, quads: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    轻量后处理：把明显不合格的 aspect 纠正为 '_'，并做枚举校验/去重/排序。
    这一步对 strict F1 提升很明显，且不耗GPU。
    """
    cleaned: List[Dict[str, str]] = []
    seen: set[Tuple[str, str, str, str]] = set()

    for q in quads:
        a = norm(q.get("aspect", ""))
        o = norm(q.get("opinion", ""))
        c = norm(q.get("category", ""))
        p = norm(q.get("polarity", ""))

        # 枚举校验
        if c not in CATEGORY_SET or p not in POLARITY_SET:
            continue
        if o == "" or o == "_":
            continue

        # opinion必须是原文子串，否则丢弃（你这项几乎都满足）
        if o not in review:
            continue

        # aspect规则：必须子串；若模型给了类别词/或非子串 -> 强制改为 "_"
        if a == "" or a == "_":
            a = "_"
        else:
            if (a in CATEGORY_SET) or (a not in review):
                a = "_"

        key = (a, o, c, p)
        if key in seen:
            continue
        seen.add(key)

        cleaned.append({"aspect": a, "opinion": o, "category": c, "polarity": p})

    # 排序：按 opinion 首次出现位置；再按 aspect（"_" 放最后）
    def sort_key(item: Dict[str, str]):
        o = item["opinion"]
        pos = review.find(o)
        a = item["aspect"]
        a_key = (1, "") if a == "_" else (0, a)
        return (pos if pos >= 0 else 10**9, a_key)

    cleaned.sort(key=sort_key)
    return cleaned


def main():
    with open(DEV_JSON, "r", encoding="utf-8") as f:
        dev_data = json.load(f)

    # 你原来的 few-shot 太长且带 start/end，我这里给一个更“干净”的 few-shot 示例写法
    few_shots = [
        {
            "review": "很好，遮暇功能差一些，总体还不错",
            "labels": [
                {"aspect": "_", "opinion": "很好", "category": "整体", "polarity": "正面"},
                {"aspect": "遮暇功能", "opinion": "差一些", "category": "功效", "polarity": "负面"},
                {"aspect": "_", "opinion": "还不错", "category": "整体", "polarity": "正面"},
            ],
        },
        {
            "review": "活动价很是划算，买一送一共60片才花八十五块，天天用都不心疼啊",
            "labels": [
                {"aspect": "_", "opinion": "很是划算", "category": "价格", "polarity": "正面"},
            ],
        },
        {
            "review": "还不错，有两个替换装味道香，但不刺鼻，可能有点儿掉妆",
            "labels": [
                {"aspect": "_", "opinion": "还不错", "category": "整体", "polarity": "正面"},
                {"aspect": "味道", "opinion": "香", "category": "气味", "polarity": "正面"},
                {"aspect": "味道", "opinion": "不刺鼻", "category": "气味", "polarity": "正面"},
                {"aspect": "_", "opinion": "有点儿掉妆", "category": "使用体验", "polarity": "负面"},
            ],
        },
    ]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    results = []
    for i, item in enumerate(dev_data, start=1):
        rid = item["id"]
        review = item["review"]

        messages = build_messages(review, few_shots=few_shots)
        prompt_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = tokenizer(prompt_text, return_tensors="pt")

        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=1600,     # 适当降低，减少“发散”
                do_sample=False,
                repetition_penalty=1.05,
                eos_token_id=tokenizer.eos_token_id,
            )

        input_len = inputs["input_ids"].shape[-1]
        gen_text = tokenizer.decode(gen_ids[0][input_len:], skip_special_tokens=True).strip()

        raw_quads = extract_json_array(gen_text)
        pred_list = postprocess_quads(review, raw_quads)

        results.append(
            {
                "id": rid,
                "review": review,
                "prediction_raw": gen_text,
                "prediction": pred_list,
            }
        )

        if i % 20 == 0:
            print(f"Processed {i}/{len(dev_data)}")

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Done. Saved to {OUT_JSON}")


if __name__ == "__main__":
    main()