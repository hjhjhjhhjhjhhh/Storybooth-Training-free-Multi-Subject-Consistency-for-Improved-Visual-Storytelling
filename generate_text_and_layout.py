import os
import json
import torch
from huggingface_hub import login, snapshot_download
from diffusers.schedulers import DPMSolverMultistepScheduler
from StoryBooth import RegionalDiffusionXLPipeline
from mllms.mllm import local_llm
import yaml
import json
from pathlib import Path

# ====== initial ======
JSON_PATH = "datasets/scene_prompts_output.json"
LLM_OUTPUT_PATH = "llm_outputs.json"

llm_model_path = "meta-llama/Meta-Llama-3.1-8B-Instruct"
#llm_model_path = "meta-llama/Llama-2-13b-chat-hf"
sd_model_path = "stabilityai/stable-diffusion-xl-base-1.0"

index_key = "1" #only run the first index
output_dir = f"output/{index_key}"
os.makedirs(output_dir, exist_ok=True)

login(token="") #放huggingface token, 不要push token到github上

# ====== loading prompt ======
with open(JSON_PATH, "r", encoding="utf-8") as f:
    scenes = json.load(f)

scene_list = scenes[index_key]


def unload_gpu():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


def patch_rope_scaling(model_id: str):
    """
    Auto-fix rope_scaling for Transformers<=4.36
    Convert Llama-3/Llama-3.1 rope_scaling to:
        {"type":"dynamic", "factor":X}
    """
    print(f"\n🔍 Locating model: {model_id}")

    # Download or locate local snapshot
    model_path = snapshot_download(model_id)
    config_path = Path(model_path) / "config.json"

    print(f"📄 config.json path: {config_path}")

    if not config_path.exists():
        raise FileNotFoundError("config.json not found!")

    # Load config.json
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # Check if rope_scaling exists
    rope = cfg.get("rope_scaling", None)
    if rope is None:
        print("✔ No rope_scaling found. Nothing to patch.")
        return

    print("\n🧩 Before patch (original rope_scaling):")
    print(json.dumps(rope, indent=4))

    factor = rope.get("factor", 1.0)

    cfg["rope_scaling"] = {
        "type": "dynamic",  # required
        "factor": factor    # preserve factor
    }

    # Save back
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=4, ensure_ascii=False)

    print("\nAfter patch (transfomers-4.36 compatible rope_scaling):")
    print(json.dumps(cfg["rope_scaling"], indent=4))

    print("\nPatch complete! You can now load the model normally.\n")

# ============================================================
# 逐個 prompt 丟給 LLM → 存到 JSON 
# ============================================================

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
            para_dict = local_llm(scene_prompt, model_path=llm_model_path)
            break  

        except ValueError as e:
            print(f"    LLM ValueError: {e}")
            print("    → Applying rope scaling patch...")
            patch_rope_scaling("meta-llama/Meta-Llama-3.1-8B-Instruct")

            attempt += 1
            unload_gpu()
            print("    [VRAM cleared before retry]")

        except Exception as e:
            print(f"    LLM other error: {e}")
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

    unload_gpu()
    print("[LLM Cleared VRAM]")


with open(LLM_OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(llm_outputs, f, indent=2, ensure_ascii=False)

print(f"\nLLM outputs saved → {LLM_OUTPUT_PATH}")
