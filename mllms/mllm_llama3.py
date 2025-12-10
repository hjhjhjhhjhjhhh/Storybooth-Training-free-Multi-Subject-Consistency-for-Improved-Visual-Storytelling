import requests
import json
import os
import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM

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


# ---------------------------------------------------------
# LOCAL LLM VERSION FOR LLAMA-3.1
# ---------------------------------------------------------
def local_llm(prompt, model_path=None):
    model_id = model_path or "meta-llama/Meta-Llama-3.1-8B-Instruct"
    print("Using model:", model_id)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
    )

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

    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    enc = tokenizer(formatted, return_tensors="pt").to(model.device)

    print("waiting for LLM response")

    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=1024,
            eos_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(out[0], skip_special_tokens=True)

    # remove prompt from output
    if decoded.startswith(formatted):
        decoded = decoded[len(formatted):]

    return get_params_dict(decoded)



# ---------------------------------------------------------
# Extract "Final split ratio" + "Regional Prompt"
# ---------------------------------------------------------
def get_params_dict(output_text):
    response = output_text

    # Find the last Final split ratio
    split_ratio_matches = re.findall(r"Final split ratio:\s*([\d.,;]+)", response)
    final_split_ratio = split_ratio_matches[-1] if split_ratio_matches else None
    print("Final split ratio:", final_split_ratio)

    # Find the last Regional Prompt
    prompt_matches = re.findall(
        r"Regional Prompt:\s*(.*?)(?:\n\s*\n|$)",
        response,
        re.DOTALL
    )
    regional_prompt = prompt_matches[-1].strip() if prompt_matches else None
    print("Regional Prompt:", regional_prompt)
    

    return {
        "Final split ratio": final_split_ratio,
        "Regional Prompt": regional_prompt
    }
