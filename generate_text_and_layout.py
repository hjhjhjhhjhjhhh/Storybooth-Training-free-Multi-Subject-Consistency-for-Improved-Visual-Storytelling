import os
import json
import torch
from huggingface_hub import login
from diffusers.schedulers import DPMSolverMultistepScheduler
from RegionalDiffusion_xl import RegionalDiffusionXLPipeline
from mllm import local_llm
import yaml
import json
from pathlib import Path
# ====== initial ======
JSON_PATH = "scene_prompts_output.json"
LLM_OUTPUT_PATH = "llm_outputs.json"

llm_model_path = "meta-llama/Llama-2-13b-chat-hf"
sd_model_path = "stabilityai/stable-diffusion-xl-base-1.0"

index_key = "1" #only run the first index
output_dir = f"output/{index_key}"
os.makedirs(output_dir, exist_ok=True)

login(token="") #放huggingface token, 不要push token到github上

# ====== loading prompt ======


def unload_gpu():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


def generate_scene_prompts(doc):
    general_prompt = doc["general_prompt"]
    style = doc["style"]
    prompts_array = doc["prompts_array"]

    scene_prompts = []
    for p in prompts_array:
        scene_prompts.append(f"{style} {general_prompt}, {p}")
    return scene_prompts


def main():
    input_path = Path("input_dataset_multi.yaml")   #Yaml file, use input_dataset_{multi or single}.yaml
    output_path = Path("scene_prompts_output.json")

    # Read YAML
    with open(input_path, "r", encoding="utf-8") as f:
        docs = list(yaml.safe_load_all(f))

    output_dict = {}

    for doc in docs:
        index = str(doc["index"])  
        output_dict[index] = generate_scene_prompts(doc)

    # output JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_dict, f, ensure_ascii=False, indent=2)

    print(f"Saved to {output_path}")
    llm_outputs = {}

    # ============================================================
    # 逐個 prompt 丟給 LLM → 存到 JSON 
    # ============================================================
    print("\n========== Running LLM ==========\n")
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        scenes = json.load(f)
    #目前只先跑第一個index，index_key = 1
    scene_list = scenes[index_key]
    for i, scene_prompt in enumerate(scene_list):

        print(f"\n---- LLM Step {i} ----")
        print("scene prompt:", scene_prompt)

        try:
            para_dict = local_llm(scene_prompt, model_path=llm_model_path)
        except Exception as e:
            print("LLM error:", e) #TODO:之後要改成error就該index重來
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


if __name__ == "__main__":
    main()









