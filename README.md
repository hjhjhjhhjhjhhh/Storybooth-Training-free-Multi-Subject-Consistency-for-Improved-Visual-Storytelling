# Unofficial implementation of Storybooth-Training-free-Multi-Subject-Consistency-for-Improved-Visual-Storytelling

## Environment setup
### 1. Install Dependencies
```bash
git clone https://github.com/YangLing0818/RPG-DiffusionMaster
cd RPG-DiffusionMaster
conda create -n RPG python==3.9
conda activate RPG
pip install -r requirements.txt
pip install protobuf
pip install transformers_stream_generator
```

### 2. Create Llama access token on huggingface  
- Visit https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
 and follow the instructions there to submit your contact information and request access to the Llama model. 

- Visit https://huggingface.co/settings/tokens, create a new token and check the permission _`Read access to contents of all public gated repos you can access`_. Make sure to store the token value `hf_XXX`.

- You are able to access the model if you see _`You have been granted access to this model`_ on the model page.

- Add the following code in your script when you need to access the model.
```python
from huggingface_hub import login
login(token="hf_XXX")
```

## How to run
### 0. Add your huggingface token
- In `generate_text_and_layout.py` and `output_image.py`, you need to add your huggingface token.
- **Do not push your token to GitHub!**
### 1. Create scene_prompts
```bash
python generate_text_and_layout.py
```
### 1-1. Use llama3
- Change `import mllms.mllm` to import `mllms.mllm_llama3`.
- Set the model to `meta-llama/Meta-Llama-3.1-8B-Instruct`.
- Remember to apply for access first at: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/discussions/15

- If you encounter the error `cannot import name 'cached_download' from 'huggingface_hub'`, go to the reported location, delete all occurrences of `cached_download`, and replace them with `hf_hub_download`.
### 2. Generate the images
```bash
python output_image.py
```
- Right now, the default setting only generates the image for index = 1. Later on, I might change it so that each index has its own folder containing the LLM output JSON file and its images.
## TODO
- [x] Intra-image self-attention bounding
- [ ] Inter-image self-attention bounding
- [ ] Cross-frame token merging
- [ ] Early negative token unmerging




