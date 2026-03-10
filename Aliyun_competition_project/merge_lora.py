import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL_DIR = "/root/lanyun-tmp/Qwen3-8B/Qwen3-8B"
LORA_DIR = "/root/LLaMA-Factory/acos_lora_output"
MERGED_DIR = "/root/lanyun-pub/acos_lora_merged"

os.makedirs(MERGED_DIR, exist_ok=True)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_DIR, trust_remote_code=True)

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_DIR,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(base_model, LORA_DIR)

print("Merging adapter into base weights...")
merged = model.merge_and_unload()

print("Saving merged model...")
merged.save_pretrained(
    MERGED_DIR,
    safe_serialization=True,
    max_shard_size="4GB"
)
tokenizer.save_pretrained(MERGED_DIR)

print("Done. Merged model saved to:", MERGED_DIR)