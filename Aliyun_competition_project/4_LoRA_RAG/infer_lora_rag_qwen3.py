import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import re
import pickle
from typing import Any, Dict, List, Optional, Tuple, Set

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
)
from peft import PeftModel

# =========================
# Config
# =========================
DEV_JSON = r"D:\Aliyun_competition_project\data\json_data\dev.json"
OUT_JSONL = r"D:\Aliyun_competition_project\data\json_data\pred_dev_lora_rag.jsonl"

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
    "价格", "物流", "包装", "味道", "补水效果", "活动", "快递", "补水", "香味", "赠品", "保湿效果", "速度", "性价比",
    "遮瑕效果", "服务态度", "气味", "送货", "服务", "隔离效果", "精华液",
    "盒子", "瓶子", "密封", "外包装",
    "配送", "发货", "运输",
    "价", "优惠", "划算",
    "味", "刺鼻",
    "日期", "生产日期", "保质期", "新鲜", "新鲜度",
    "成分", "配方",
    "尺寸", "大小", "规格",
    "客服", "售后",
    "功效", "效果", "作用", "保湿", "遮瑕", "隔离",
    "使用体验", "体验", "肤感", "质地", "吸收", "油腻", "清爽", "掉妆",
    "真伪", "真假", "正品",
    "颜色", "色号",
    "容量", "分量",
]

# =========================
# RAG (BM25 + Rerank) Config
# =========================
USE_RAG = True

RAG_ASSETS_DIR = r"D:\Aliyun_competition_project\data\rag_assets"
TRAIN_META_JSON = os.path.join(RAG_ASSETS_DIR, "train_meta.json")
BM25_PKL = os.path.join(RAG_ASSETS_DIR, "bm25.pkl")

# BM25 先召回 20
BM25_TOPN = 20

# 最终 few-shot 改为 5
FEWSHOT_TOPK = 5

RAG_EXCLUDE_SAME_ID = True
RAG_DEDUP_BY_ID = True
RAG_MIN_QUERY_LEN = 2

# few-shot 过滤：去掉“全是 aspect='_'”的案例
FILTER_ALL_UNDERSCORE_SHOTS = True

# reranker
USE_RERANK = True
# 有网可直接用 repo id；无网请改成本地路径
RERANKER_MODEL = "bge-reranker-v2-m3"
RERANK_BATCH_SIZE = 8
RERANK_MAX_LENGTH = 512

# 如果你想保留“固定 few_shots”作为兜底，可设 True
USE_STATIC_FEWSHOT_FALLBACK = False

static_few_shots = [
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


# =========================
# Utils
# =========================
def norm(s: Any) -> str:
    return str(s).strip() if s is not None else ""

def strip_think(text: str) -> str:
    return re.sub(r"<think>[\s\S]*?</think>", "", text).strip()

def safe_json_loads(s: str):
    try:
        return json.loads(s)
    except Exception:
        return None

def find_first_complete_json_array(text: str) -> Optional[str]:
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
    return None

def salvage_objects_from_truncated_text(text: str) -> List[Dict[str, Any]]:
    objs = []
    for m in re.finditer(r"\{[^{}]*\}", text):
        obj = safe_json_loads(m.group(0))
        if isinstance(obj, dict):
            objs.append(obj)
    return objs


# =========================
# RAG helpers
# =========================
def _normalize_text(s: str) -> str:
    s = "" if s is None else str(s)
    return re.sub(r"\s+", " ", s).strip()

def _tokenize_char(text: str) -> List[str]:
    text = _normalize_text(text)
    return [ch for ch in text if ch.strip()]

def _tokenize_jieba(text: str) -> List[str]:
    text = _normalize_text(text)
    import jieba
    return [t for t in jieba.lcut(text, cut_all=False) if t.strip()]

def load_rag_assets():
    if not USE_RAG:
        return None, None, None

    if (not os.path.exists(TRAIN_META_JSON)) or (not os.path.exists(BM25_PKL)):
        raise FileNotFoundError(
            f"RAG files not found.\nTRAIN_META_JSON={TRAIN_META_JSON}\nBM25_PKL={BM25_PKL}\n"
            "Please build BM25 index first."
        )

    with open(TRAIN_META_JSON, "r", encoding="utf-8") as f:
        train_meta = json.load(f)

    with open(BM25_PKL, "rb") as f:
        obj = pickle.load(f)

    if isinstance(obj, dict):
        bm25 = obj.get("bm25", None)
        tok_name = obj.get("tokenizer", "jieba")
        id_list = obj.get("id_list", None)
    else:
        bm25 = obj
        tok_name = "jieba"
        id_list = None

    if bm25 is None:
        raise ValueError("BM25_PKL does not contain BM25 object.")

    tok_fn = _tokenize_jieba if tok_name == "jieba" else _tokenize_char

    if id_list is None:
        id_list = [int(x.get("id", i)) for i, x in enumerate(train_meta)]

    if len(train_meta) != len(id_list):
        raise ValueError(f"train_meta size {len(train_meta)} != bm25 id_list size {len(id_list)}")

    return train_meta, bm25, (id_list, tok_fn)

def _clean_labels_4fields(labels: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    out = []
    for q in labels or []:
        out.append({
            "aspect": (str(q.get("aspect", "")).strip() or "_"),
            "opinion": str(q.get("opinion", "")).strip(),
            "category": str(q.get("category", "")).strip(),
            "polarity": str(q.get("polarity", "")).strip(),
        })
    return out

def _is_all_underscore_case(labels: List[Dict[str, str]]) -> bool:
    valid = [q for q in labels if q.get("opinion")]
    if not valid:
        return True
    return all((q.get("aspect", "_").strip() == "_") for q in valid)

def bm25_retrieve_candidates(
    train_meta: List[Dict[str, Any]],
    bm25,
    tok_pack,
    query_review: str,
    topn: int = 20,
    exclude_id: Optional[int] = None,
) -> List[Dict[str, Any]]:
    query_review = _normalize_text(query_review)
    if len(query_review) < RAG_MIN_QUERY_LEN:
        return []

    id_list, tok_fn = tok_pack
    q_tokens = tok_fn(query_review)
    if not q_tokens:
        return []

    scores = bm25.get_scores(q_tokens)
    idx_score = list(enumerate(scores))
    idx_score.sort(key=lambda x: float(x[1]), reverse=True)

    hits = []
    seen_ids = set()

    for idx, sc in idx_score[:topn]:
        ex = train_meta[idx]
        tid = int(ex.get("id", id_list[idx]))

        if RAG_EXCLUDE_SAME_ID and (exclude_id is not None) and (tid == int(exclude_id)):
            continue
        if RAG_DEDUP_BY_ID and tid in seen_ids:
            continue
        seen_ids.add(tid)

        labels = _clean_labels_4fields(ex.get("labels", []))

        if FILTER_ALL_UNDERSCORE_SHOTS and _is_all_underscore_case(labels):
            continue

        hits.append({
            "id": tid,
            "review": _normalize_text(ex.get("review", "")),
            "labels": labels,
            "bm25_score": float(sc),
        })

    return hits

# =========================
# Reranker
# =========================
def load_reranker():
    if not USE_RERANK:
        return None, None, None

    rerank_tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL, trust_remote_code=True)
    rerank_model = AutoModelForSequenceClassification.from_pretrained(
        RERANKER_MODEL,
        trust_remote_code=True,
    )

    if torch.cuda.is_available():
        rerank_model = rerank_model.to("cuda")
    rerank_model.eval()

    return rerank_tokenizer, rerank_model, ("cuda" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def rerank_hits(
    query: str,
    hits: List[Dict[str, Any]],
    rerank_pack,
    topk: int = 5,
) -> List[Dict[str, Any]]:
    if (not USE_RERANK) or (rerank_pack is None) or (not hits):
        return hits[:topk]

    rerank_tokenizer, rerank_model, rerank_device = rerank_pack

    pairs = [[query, h["review"]] for h in hits]
    all_scores = []

    for start in range(0, len(pairs), RERANK_BATCH_SIZE):
        batch_pairs = pairs[start:start + RERANK_BATCH_SIZE]
        enc = rerank_tokenizer(
            batch_pairs,
            padding=True,
            truncation=True,
            max_length=RERANK_MAX_LENGTH,
            return_tensors="pt"
        )
        enc = {k: v.to(rerank_device) for k, v in enc.items()}
        outputs = rerank_model(**enc)
        logits = outputs.logits.view(-1).detach().float().cpu().tolist()
        all_scores.extend(logits)

    for h, sc in zip(hits, all_scores):
        h["rerank_score"] = float(sc)

    hits = sorted(hits, key=lambda x: x["rerank_score"], reverse=True)
    return hits[:topk]


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
        user_parts.append("下面是一些相似案例（仅供参考抽取方式）：")
        user_parts.append("请注意：这些案例只用于学习抽取格式、属性-观点配对方式、类别划分和极性判断，不要模仿案例内容；你的输出必须严格依据当前评论原文，不能照抄案例中的词语。")
        for ex in few_shots:
            user_parts.append(f"评论：{ex['review']}\n输出：{json.dumps(ex['labels'], ensure_ascii=False)}\n")
    user_parts.append("现在请对下面评论进行抽取，只输出 JSON：")
    user_parts.append(f"评论：{review}")

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n".join(user_parts)},
    ]

def build_messages_retry(review: str) -> List[Dict[str, str]]:
    return build_messages_main(review, few_shots=None)


# =========================
# Robust extractor
# =========================
def extract_quads(text: str) -> List[Dict[str, Any]]:
    text = strip_think(text)

    arr_snip = find_first_complete_json_array(text)
    if arr_snip:
        obj = safe_json_loads(arr_snip)
        if isinstance(obj, list):
            return obj

    obj = safe_json_loads(text)
    if isinstance(obj, list):
        return obj

    return salvage_objects_from_truncated_text(text)


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

    for trig in sorted(ASPECT_TRIGGERS, key=len, reverse=True):
        if trig in left:
            return trig

    if category == "物流":
        for w in ["物流", "快递", "送货", "配送", "发货", "速度"]:
            if w in left:
                return w

    if category == "价格":
        for w in ["性价比", "价格", "活动", "优惠", "划算"]:
            if w in left:
                return w

    if category == "气味":
        for w in ["味道", "香味", "气味"]:
            if w in left:
                return w

    if category == "服务":
        for w in ["服务态度", "服务", "客服", "售后"]:
            if w in left:
                return w

    m = re.search(r"([\u4e00-\u9fa5]{1,6}效果)", left)
    if m:
        return m.group(1)

    return None

def validate_and_postprocess(review: str, quads: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    seen: Set[Tuple[str, str, str, str]] = set()
    out: List[Dict[str, str]] = []

    for q in quads:
        qn = normalize_quad(q)
        a, o, c, p = qn["aspect"], qn["opinion"], qn["category"], qn["polarity"]

        if c not in CATEGORY_SET or p not in POLARITY_SET:
            continue

        if not o or o == "_" or o not in review:
            continue

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
    prompt_text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = tokenizer(prompt_text, return_tensors="pt")

    # 关键：把输入放到“输入嵌入层”的设备上
    try:
        input_device = model.get_input_embeddings().weight.device
    except Exception:
        try:
            input_device = model.base_model.get_input_embeddings().weight.device
        except Exception:
            input_device = next(model.parameters()).device

    inputs = {k: v.to(input_device) for k, v in inputs.items()}

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

    # RAG 资源加载
    train_meta, bm25, tok_pack = load_rag_assets()
    if USE_RAG:
        print(f"[RAG] Loaded train_meta={len(train_meta)} (bm25_topN={BM25_TOPN}, final_topK={FEWSHOT_TOPK})")
        print(f"[RAG] TRAIN_META_JSON={TRAIN_META_JSON}")
        print(f"[RAG] BM25_PKL={BM25_PKL}")

    rerank_pack = load_reranker() if USE_RERANK else None
    if USE_RERANK:
        print(f"[RERANK] Loaded reranker: {RERANKER_MODEL}")

    # LoRA(base+adapter) paths
    BASE_MODEL = r"/root/lanyun-tmp/Qwen3-8B/Qwen3-8B"
    LORA_DIR = r"/root/LLaMA-Factory/acos_lora_output"

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(base_model, LORA_DIR)
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

            # 1) BM25 召回
            few_shots = None
            hits = []
            if USE_RAG:
                hits = bm25_retrieve_candidates(
                    train_meta=train_meta,
                    bm25=bm25,
                    tok_pack=tok_pack,
                    query_review=review,
                    topn=BM25_TOPN,
                    exclude_id=rid if RAG_EXCLUDE_SAME_ID else None,
                )

                # 2) rerank -> topK
                hits = rerank_hits(
                    query=review,
                    hits=hits,
                    rerank_pack=rerank_pack,
                    topk=FEWSHOT_TOPK,
                )

                # 给 prompt 的 few-shot
                few_shots = [{"review": x["review"], "labels": x["labels"], "id": x["id"]} for x in hits]

            # 可选兜底
            if (not few_shots) and USE_STATIC_FEWSHOT_FALLBACK:
                few_shots = static_few_shots

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
                "rag_fewshot_ids": [int(x["id"]) for x in (few_shots or [])],
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