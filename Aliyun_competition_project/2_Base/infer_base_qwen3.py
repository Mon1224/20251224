import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import re
from typing import Any, Dict, List, Optional, Tuple, Set

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# =========================
# Config
# =========================
MODEL_ID = "Qwen/Qwen3-8B"  # 或 "/root/autodl-tmp/Qwen3-8B/Qwen3-8B"
DEV_JSON = r"D:\Aliyun_competition_project\data\json_data\dev.json"
OUT_JSONL = r"D:\Aliyun_competition_project\data\json_data\pred_dev_base.jsonl"

CATEGORY_LIST = ["包装","成分","尺寸","服务","功效","价格","气味","使用体验","物流","新鲜度","真伪","整体","其他"]
POLARITY_LIST = ["正面", "中性", "负面"]
CATEGORY_SET = set(CATEGORY_LIST)
POLARITY_SET = set(POLARITY_LIST)

MAX_NEW_TOKENS_MAIN = 1600
MAX_NEW_TOKENS_RETRY = 1200

RETRY_IF_EMPTY = True
RETRY_MAX = 1  # 成本控制：最多重试1次

# aspect 回填：左侧窗口大小
ASPECT_BACKFILL_LEFT_WINDOW = 20

# 常见属性触发词（可按你的数据再扩展）
ASPECT_TRIGGERS = [
    "包装", "盒子", "瓶子", "密封", "外包装",
    "物流", "快递", "配送", "发货", "运输",
    "价格", "价", "活动", "优惠", "划算",
    "味道", "气味", "香味", "味", "刺鼻",
    "日期", "生产日期", "保质期", "新鲜", "新鲜度",
    "成分", "配方",
    "尺寸", "大小", "规格",
    "服务", "客服", "售后",
    "功效", "效果", "作用",
    "使用体验", "体验", "肤感", "质地", "吸收", "油腻", "清爽", "掉妆",
    "真伪", "真假", "正品",
    "颜色", "色号",
    "容量", "分量"
]


# =========================
# Utils
# =========================
def norm(s: Any) -> str:
    return str(s).strip() if s is not None else ""

def strip_think(text: str) -> str:
    # 去掉 <think>...</think> 段
    return re.sub(r"<think>[\s\S]*?</think>", "", text).strip()

def safe_json_loads(s: str):
    try:
        return json.loads(s)
    except Exception:
        return None

def find_first_complete_json_array(text: str) -> Optional[str]:
    """
    在混杂文本中找“首个完整闭合 JSON 数组字符串”。
    使用括号计数，避免正则贪婪吞噬。
    """
    start = text.find("[")
    if start == -1:
        return None

    depth = 0
    in_str = False
    esc = False

    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    return text[start:i+1]
    return None  # 未闭合，可能被截断

def salvage_objects_from_truncated_text(text: str) -> List[Dict[str, Any]]:
    """
    截断容错：尽量捡回多个完整的 {...} 对象（适用于四元组扁平对象）。
    """
    objs = []
    for m in re.finditer(r"\{[^{}]*\}", text):
        obj = safe_json_loads(m.group(0))
        if isinstance(obj, dict):
            objs.append(obj)
    return objs


# =========================
# Prompt builders (system MUST be exact)
# =========================
def build_messages_main(review: str, few_shots: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, str]]:
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
        user_parts.append("下面是一些示例：")
        for ex in few_shots:
            user_parts.append(f"评论：{ex['review']}\n输出：{json.dumps(ex['labels'], ensure_ascii=False)}\n")
    user_parts.append("现在请对下面评论进行抽取，只输出 JSON：")
    user_parts.append(f"评论：{review}")

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n".join(user_parts)},
    ]


def build_messages_retry(review: str) -> List[Dict[str, str]]:
    # 重试时仍然使用同一套 system 原文（不改），只减少 few-shot 干扰
    return build_messages_main(review, few_shots=None)


# =========================
# Robust extractor
# =========================
def extract_quads(text: str) -> List[Dict[str, Any]]:
    """
    主路径：只找首个完整 JSON 数组并解析。
    失败：尝试 direct loads（可能是纯数组）。
    最后：截断容错捡 {...}。
    """
    text = strip_think(text)

    arr_snip = find_first_complete_json_array(text)
    if arr_snip:
        obj = safe_json_loads(arr_snip)
        if isinstance(obj, list):
            return obj

    obj = safe_json_loads(text)
    if isinstance(obj, list):
        return obj

    salvaged = salvage_objects_from_truncated_text(text)
    return salvaged


# =========================
# Output governance + aspect backfill
# =========================
def normalize_quad(q: Dict[str, Any]) -> Dict[str, str]:
    return {
        "aspect": norm(q.get("aspect", "")),
        "opinion": norm(q.get("opinion", "")),
        "category": norm(q.get("category", "")),
        "polarity": norm(q.get("polarity", "")),
    }

def aspect_backfill(review: str, opinion: str, category: str) -> Optional[str]:
    pos = review.find(opinion)
    if pos < 0:
        return None

    left_start = max(0, pos - ASPECT_BACKFILL_LEFT_WINDOW)
    left = review[left_start:pos]

    # 1️⃣ 优先匹配完整触发词（长词优先）
    for trig in sorted(ASPECT_TRIGGERS, key=len, reverse=True):
        if trig in left:
            return trig

    # 2️⃣ 按 category 做定向回填
    if category == "物流":
        for w in ["物流", "快递", "送货", "配送", "发货", "速度"]:
            if w in left:
                return w

    if category == "价格":
        for w in ["性价比", "价格", "活动"]:
            if w in left:
                return w

    if category == "气味":
        for w in ["味道", "香味", "气味"]:
            if w in left:
                return w

    if category == "服务":
        for w in ["服务态度", "服务", "客服"]:
            if w in left:
                return w

    # 3️⃣ 泛规则：匹配 “X效果”
    m = re.search(r"([\u4e00-\u9fa5]{1,6}效果)", left)
    if m:
        return m.group(1)

    return None

def validate_and_postprocess(review: str, quads: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    校验：枚举/子串/字段完整性
    aspect：若非子串/或等于category词，则尝试回填；回填失败 -> "_"
    去重：四字段
    排序：按 opinion 首次出现位置；同位置按 aspect（"_" 最大）
    """
    seen: Set[Tuple[str, str, str, str]] = set()
    out: List[Dict[str, str]] = []

    for q in quads:
        qn = normalize_quad(q)
        a, o, c, p = qn["aspect"], qn["opinion"], qn["category"], qn["polarity"]

        # 枚举校验
        if c not in CATEGORY_SET or p not in POLARITY_SET:
            continue

        # opinion 子串校验
        if not o or o == "_" or o not in review:
            continue

        # aspect 处理：
        # 1) 空 / "_" => "_"
        # 2) category词当aspect => 回填 or "_"
        # 3) 非子串 => 回填 or "_"
        if not a or a == "_":
            a = "_"
        else:
            if (a in CATEGORY_SET) or (a not in review):
                filled = aspect_backfill(review, o, c)
                a = filled if filled else "_"

        key = (a, o, c, p)
        if key in seen:
            continue
        seen.add(key)
        out.append({"aspect": a, "opinion": o, "category": c, "polarity": p})

    def sort_key(item: Dict[str, str]):
        pos = review.find(item["opinion"])
        pos = pos if pos >= 0 else 10**9
        a = item["aspect"]
        # "_" 视为最大
        a_key = (1, "") if a == "_" else (0, a)
        return (pos, a_key)

    out.sort(key=sort_key)
    return out


# =========================
# Incremental IO (jsonl)
# =========================
def load_done_ids(jsonl_path: str) -> Set[int]:
    done = set()
    if not os.path.exists(jsonl_path):
        return done
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if "id" in obj:
                    done.add(int(obj["id"]))
            except Exception:
                continue
    return done


# =========================
# Inference
# =========================
def run_generate(model, tokenizer, messages: List[Dict[str, str]], max_new_tokens: int) -> str:
    prompt_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = tokenizer(prompt_text, return_tensors="pt")

    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    with torch.no_grad():
        gen_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.05,
            eos_token_id=tokenizer.eos_token_id,
        )

    input_len = inputs["input_ids"].shape[-1]
    return tokenizer.decode(gen_ids[0][input_len:], skip_special_tokens=True).strip()


def main():
    with open(DEV_JSON, "r", encoding="utf-8") as f:
        dev_data = json.load(f)

    done_ids = load_done_ids(OUT_JSONL)
    print(f"[Resume] Already done: {len(done_ids)} ids")

    # few-shot：保持少量、干净（不含 start/end）
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
            "labels": [{"aspect": "_", "opinion": "很是划算", "category": "价格", "polarity": "正面"}],
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

    out_dir = os.path.dirname(OUT_JSONL)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    retry_used = 0
    processed = 0

    with open(OUT_JSONL, "a", encoding="utf-8") as fout:
        for item in dev_data:
            rid = int(item["id"])
            review = str(item["review"])

            if rid in done_ids:
                continue

            # main
            messages = build_messages_main(review, few_shots=few_shots)
            gen_text = run_generate(model, tokenizer, messages, MAX_NEW_TOKENS_MAIN)

            raw_quads = extract_quads(gen_text)
            pred = validate_and_postprocess(review, raw_quads)

            # retry: only if empty
            if RETRY_IF_EMPTY and len(pred) == 0:
                for _ in range(RETRY_MAX):
                    retry_messages = build_messages_retry(review)
                    gen_text2 = run_generate(model, tokenizer, retry_messages, MAX_NEW_TOKENS_RETRY)
                    raw_quads2 = extract_quads(gen_text2)
                    pred2 = validate_and_postprocess(review, raw_quads2)
                    if len(pred2) > 0:
                        gen_text = gen_text2
                        pred = pred2
                        retry_used += 1
                        break

            record = {
                "id": rid,
                "review": review,
                "prediction_raw": gen_text,
                "prediction": pred,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()

            done_ids.add(rid)
            processed += 1
            if processed % 20 == 0:
                print(f"Processed +{processed} (retry_used={retry_used})")

    print(f"Done. Written jsonl to {OUT_JSONL}")
    print(f"Retry used on {retry_used} samples.")


if __name__ == "__main__":
    main()