import os
import gc
import json
import torch
from huggingface_hub import login
from mllm_llama3 import local_llm, load_local_llm
import yaml
import json
from pathlib import Path
from tokens import HUGGINGFACE_TOKEN

# ====== initial ======
JSON_PATH = "scene_prompts_output.json"
LLM_OUTPUT_PATH = "llm_outputs.json"

llm_model_path = "meta-llama/Meta-Llama-3.1-8B-Instruct"
#llm_model_path = "meta-llama/Llama-2-13b-chat-hf"
sd_model_path = "stabilityai/stable-diffusion-xl-base-1.0"

index_key = "1" #only run the first index
output_dir = f"output/{index_key}"
os.makedirs(output_dir, exist_ok=True)

login(token=HUGGINGFACE_TOKEN) #放huggingface token, 不要push token到github上

# ====== loading prompt ======
with open(JSON_PATH, "r", encoding="utf-8") as f:
    scenes = json.load(f)

scene_list = scenes[index_key]


def unload_gpu():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    gc.collect() 


# ============================================================
# 逐個 prompt 丟給 LLM → 存到 JSON 
# ============================================================

tokenizer, model = load_local_llm(llm_model_path)

llm_outputs = {}

print("\n========== PHASE 1: Running LLM ==========\n")

for i, scene_prompt in enumerate(scene_list):

    print(f"\n---- LLM Step {i} ----")
    print("scene prompt:", scene_prompt)

    # retry 
    max_retry = 5
    attempt = 0
    para_dict = None

    while attempt < max_retry:
        try:
            print(f"  → Attempt {attempt + 1}/{max_retry}")
            para_dict = local_llm(scene_prompt, tokenizer, model)
            break  
        except Exception as e:
            print(f"    LLM error: {e}")
            attempt += 1
            unload_gpu() 
            print("    [VRAM cleared before retry]")

    if para_dict is None:
        print(f"Failed after {max_retry} attempts. Skipping this prompt.")
        continue

    llm_outputs[str(i)] = {
        "regional_prompt": para_dict["Regional Prompt"],
        "split_ratio": para_dict["Final split ratio"],
        "base_prompt": scene_prompt,
    }

    
    del para_dict
    unload_gpu()
    print("[LLM Cleared VRAM]")


with open(LLM_OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(llm_outputs, f, indent=2, ensure_ascii=False)

print(f"\nLLM outputs saved → {LLM_OUTPUT_PATH}")
