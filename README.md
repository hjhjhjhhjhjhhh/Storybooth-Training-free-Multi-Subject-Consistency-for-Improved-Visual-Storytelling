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

## TODO
- [x] Intra-image self-attention bounding
- [ ] Inter-image self-attention bounding
- [ ] Cross-frame token merging
- [ ] Early negative token unmerging




