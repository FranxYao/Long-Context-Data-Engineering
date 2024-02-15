
# Long-Context Data Engineering

<p align="center" width="100%">
<a ><img src="assets/logo.jpg" alt="logo" style="width: 80%; min-width: 300px; display: block; margin: auto;"></a>
</p>

ChatGPT-4 Dalle-3 Prompt: "Draw a carton style logo showing a very very long paper"
<p align="center">
    🤗 <a href="https://huggingface.co/yaofu/llama-2-7b-80k" target="_blank">HF Repo</a> • 📃 <a href="https://github.com/FranxYao/Long-Context-Data-Engineering/assets/long_context_data_engineering.pdf" target="_blank">Paper</a>
</p>

Implementation of paper:
* Yao Fu, Rameswar Panda, Xinyao Niu, Xiang Yue, Hannaneh Hajishirzi, Yoon Kim and Hao Peng. Feb 2024. _Data Engineering for Scaling Language Models to 128K Context_

We provide the script for
- [x] Loading and playing with the following continue pretrained checkpoint:
    - [x]  LLaMA-2 7B 80K: continue pretrained on 80K, tested on 128K
    - [ ] LLaMA-2 13B 64K: continue pretrained on 64K, tested on 128K
- [ ] Loading the preprocessed data
- [ ] Evaluating the pretrained checkpoint on Needle-in-a-HayStack
- [ ] Processing the long-context data
- [ ] Continue pretraining the model on processed long-context data

## Download the model to local 
Create a folder to download the model. 
```bash 
pip install -r requirements.txt # pytorch is not included here because we assume you have already installed pytorch
mkdir ../llama-2-7b-80k
```

Download the continue pretrained checkpoint to local 
```python 
from huggingface_hub import snapshot_download
import os

snapshot_download(repo_id='yaofu/llama-2-7b-80k',
                  local_dir='llama-2-7b-80k',
                  repo_type='model',
                  local_dir_use_symlinks=False,
                  resume_download=True)
```

We recommend you download the checkpoint to local first, instead of directly loading from HF, like the following:
```python
from transformers import AutoModelForCausalLM
# Below is slow and hard to control in a cluster
# Unless you insist, we recommend you download the model to local first!
model = AutoModelForCausalLM.from_pretrained("yaofu/llama-2-7b-80k", 
                                             use_flash_attention_2="flash_attention_2", 
                                             torch_dtype=torch.bfloat16
                                             ) 
```

## Load the continue pretrained checkpoint and play with it 
The following code requries at least 8x4090 to support 80K context. 
If you have 8x80G A100 you can make it to at least 128K

We use `tensor_parallel` implemented from [this repo](https://github.com/BlackSamorez/tensor_parallel) because it is much faster than huggingface's `device_map` and lightweight than vLLM. But it has a small bug that if your GPU memory is not large enough, it will stuck instead of through a memory overflow exception. So make sure you do have enough GPU memory.
```python 
import torch 
import tensor_parallel as tp
from transformers import AutoModelForCausalLM, AutoTokenizer
from eval.needle.utils import load_context, insert_needle

# This is the continue pretrained LLaMA 2 7B model with modified rope
model = AutoModelForCausalLM.from_pretrained("../llama-2-7b-80k",
                                             use_flash_attention_2="flash_attention_2", 
                                             torch_dtype=torch.bfloat16
                                             ) # requires about 14G disk size in $HF_HOME
model = tp.tensor_parallel(model, sharded=True)

# Construct the Needle-in-a-HayStack Prompt
needle = "\nThe best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.\n"
ctx_len = 80000 # need at least 8*4090 to run this length
context = load_context(fpath="eval/needle/PaulGrahamEssays/*.txt", ctx_len=ctx_len)
context = insert_needle(context, needle, depth=0.1)
needle_idx = context.find("The best thing to do in San Francisco is")
print("Context has %d chars, needle inserted at %d char location:\n" % (len(context), needle_idx))
print(context[needle_idx - 150: needle_idx + 150]) # look at how the needle is inserted 

prompt ="\n<|im_start|> This is a very long story book: <book> %s </book>.\n" % context
question = "What is the best thing to do in San Francisco?"
prompt += "Based on the content of the book, Question: %s\nAnswer:" % question
print(prompt) # feel the length of 100K

# Check how the model performs
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
prompt = tokenizer(prompt, return_tensors="pt")
input_ids = prompt['input_ids'].to(model.device)
print("After tokenization, there is %d tokens" % len(input_ids[0]))
with torch.no_grad():
    output_ids = model.generate(input_ids, max_new_tokens=50)
    response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
print("Response:", response.split("\n")[0])
```