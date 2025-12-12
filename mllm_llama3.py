import requests
import json
import os
import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
# from transformers import BitsAndBytesConfig # For VRAM only 12G 

# ---------------------------------------------------------
# GPT-4 API version (keep original)
# ---------------------------------------------------------
def GPT4(prompt, key):
    url = "https://api.openai.com/v1/chat/completions"

    with open('template/template.txt', 'r', encoding='utf-8') as f:
        template = f.read()

    user_textprompt = f"Caption:{prompt}\nLet's think step by step:"

    full_prompt = f"{template}\n{user_textprompt}"

    payload = json.dumps({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": full_prompt}]
    })

    headers = {
        'Accept': 'application/json',
        'Authorization': f'Bearer {key}',
        'Content-Type': 'application/json'
    }

    print('waiting for GPT-4 response')
    response = requests.post(url, headers=headers, data=payload)
    obj = response.json()
    text = obj['choices'][0]['message']['content']

    return get_params_dict(text)

def load_local_llm(model_path=None):
    model_id = model_path or "meta-llama/Meta-Llama-3.1-8B-Instruct"
    print("Using model:", model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # # For VRAM only 12G 
    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=torch.float16
    # )
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_id,
    #     device_map="cuda",
    #     quantization_config=bnb_config
    # )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    return tokenizer, model

# ---------------------------------------------------------
# LOCAL LLM VERSION FOR LLAMA-3.1
# ---------------------------------------------------------
def local_llm(prompt, tokenizer, model):
    # ----- template -----
    with open("template/template.txt", "r") as f:
        template = ''.join(f.readlines())

    # ----- build chat message -----
    system_prompt = template.strip()
    user_prompt = f"Caption: {prompt}\nLet's think step by step."

    # llama 3.1 chat formatting
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to("cuda")

    print("waiting for LLM response")

    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=1024,
            eos_token_id=tokenizer.eos_token_id
        )

    response = out[0][input_ids.shape[-1]:]
    generated_text = tokenizer.decode(response, skip_special_tokens=True)
    
    return get_params_dict(generated_text)


def get_params_dict(output_text):
    heights = re.findall(r"Row([0-9]+) \(height=([0-9.]+)\)", output_text)
    widths = re.findall(r"Region([0-9]+) \(Row([0-9]+), width=([0-9\.]+)\)", output_text)
    break_count = output_text.count('BREAK')

    rows_weights = [None] * (int(heights[-1][0]) + 1)
    cols_weights = [[] for _ in range(int(heights[-1][0]) + 1)]
    for element in heights:
        rows_weights[int(element[0])] = float(element[1])
    widths_index = 0
    prev_region = -1
    element = widths[widths_index]
    region = int(element[0])
    while widths_index < len(widths):
        while region > prev_region:
            cols_weights[int(element[1])].append(float(element[2]))
            prev_region = region
            widths_index += 1
            if widths_index >= len(widths):
                break
            element = widths[widths_index]
            region = int(element[0])
        if region < prev_region:
            prev_region = -1
            cols_weights[int(element[1])].clear()

    split_ratio = ""
    if len(rows_weights) == 1:
        split_ratio = str(cols_weights[0][0])
        for j in range(1, len(cols_weights[0])):
            split_ratio += f",{cols_weights[0][j]}"
    else:
        for i in range(len(rows_weights)):
            if i > 0:
                split_ratio += ";"
            split_ratio += str(rows_weights[i])
            for j in range(len(cols_weights[i])):
                split_ratio += f",{cols_weights[i][j]}"

    regional_prompt = re.findall(r"Regional Prompt:\s+(.+)", output_text)[-1]
    print("Final split ratio:", split_ratio)
    print("Regional Prompt:", regional_prompt)

    return {
        "Final split ratio": split_ratio,
        "Regional Prompt": regional_prompt
    }