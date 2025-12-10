import os
import json
import torch
from huggingface_hub import login
from diffusers.schedulers import DPMSolverMultistepScheduler
from StoryBooth import RegionalDiffusionXLPipeline
from mllms.mllm import local_llm

JSON_PATH = "scene_prompts_output.json"
LLM_OUTPUT_PATH = "llm_outputs.json"

llm_model_path = "meta-llama/Llama-2-13b-chat-hf"
sd_model_path = "stabilityai/stable-diffusion-xl-base-1.0"

index_key = "1"
output_dir = f"output/{index_key}"
os.makedirs(output_dir, exist_ok=True)

login(token="")#放huggingface token, 不要push token到github上
# ============================================================
# Read LLM JSON → Regional Diffusion XL → generate image
# ============================================================

def unload_gpu():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

print("\n========== PHASE 2: Generating Images ==========\n")

with open(LLM_OUTPUT_PATH, "r", encoding="utf-8") as f:
    llm_outputs = json.load(f)


for key, item in llm_outputs.items():

    regional_prompt = item["regional_prompt"]
    split_ratio = item["split_ratio"]
    base_prompt = item["base_prompt"]

    print(f"\n---- PIPE Step {key} ----")
    print("Regional Prompt:", regional_prompt)
    print("Split Ratio:", split_ratio)

    # load pipeline
    pipe = RegionalDiffusionXLPipeline.from_pretrained(
        sd_model_path,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    )
    pipe.to("cuda:1")

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        use_karras_sigmas=True
    )

    pipe.enable_xformers_memory_efficient_attention() #xformer

    img = pipe(
        prompt=regional_prompt,
        split_ratio=split_ratio,
        base_ratio=0.8,
        base_prompt=base_prompt,
        num_inference_steps=40,
        height=1024,
        width=1024,
        negative_prompt="",
        seed=1234,
        guidance_scale=9,
        batch_size=1
    ).images[0]

    save_path = f"{output_dir}/image_{key}.png"
    img.save(save_path)
    print(f"Saved → {save_path}")

    del pipe
    unload_gpu()

print("\nALL DONE.")