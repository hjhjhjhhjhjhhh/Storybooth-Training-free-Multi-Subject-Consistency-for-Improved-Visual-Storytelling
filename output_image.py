import os
import gc
import json
import torch
from huggingface_hub import login
from diffusers.schedulers import DPMSolverMultistepScheduler
from StoryBooth import RegionalDiffusionXLPipeline
# from StoryBooth_base import RegionalDiffusionPipeline
from tokens import HUGGINGFACE_TOKEN

JSON_PATH = "scene_prompts_output.json"
LLM_OUTPUT_PATH = "output/1/llm_outputs.json"

# sd_model_path = "stable-diffusion-v1-5/stable-diffusion-v1-5" # For VRAM only 12G 
sd_model_path = "stabilityai/stable-diffusion-xl-base-1.0"

index_key = "1"
output_dir = f"output/{index_key}"
os.makedirs(output_dir, exist_ok=True)

login(token=HUGGINGFACE_TOKEN)#放huggingface token, 不要push token到github上
# ============================================================
# Read LLM JSON → Regional Diffusion XL → generate image
# ============================================================

def unload_gpu():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    gc.collect() 

print("\n========== PHASE 2: Generating Images ==========\n")

with open(LLM_OUTPUT_PATH, "r", encoding="utf-8") as f:
    llm_outputs = json.load(f)

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

# For VRAM only 12G 
# pipe = RegionalDiffusionPipeline.from_pretrained(
#     sd_model_path,
#     torch_dtype=torch.float16,
#     use_safetensors=True,
#     variant="fp16",
# )
# pipe.to("cuda")

# pipe.scheduler = DPMSolverMultistepScheduler.from_config(
#     pipe.scheduler.config,
#     use_karras_sigmas=True
# )

# pipe.enable_xformers_memory_efficient_attention() #xformer

rpl = []
srl = []
bpl = []

for key, item in llm_outputs.items():

    regional_prompt = item["regional_prompt"]
    split_ratio = item["split_ratio"]
    base_prompt = item["base_prompt"]

    print(f"\n---- PIPE Step {key} ----")
    print("Regional Prompt:", regional_prompt)
    print("Split Ratio:", split_ratio)

    rpl.append(regional_prompt)
    srl.append(split_ratio)
    bpl.append(base_prompt)

images = pipe(
        prompt=rpl,
        split_ratio=srl,
        base_ratio=0.8,
        base_prompt=bpl,
        num_inference_steps=50,
        height=600,
        width=600,
        negative_prompt="",
        seed=1234,
        guidance_scale=15.0,
        inter_subject_k = 2,
        subject_token = ["dog", "owl"],
        beta_d = 0.5
    ).images

K = 1
B = len(rpl)

outdir = "output7_inter"
os.makedirs(outdir, exist_ok=True)

for b in range(B):
    for k in range(K):
        idx = b * K + k
        images[idx].save(os.path.join(outdir, f"prompt{b:02d}_k{k:02d}.png"))

print("\nALL DONE.")