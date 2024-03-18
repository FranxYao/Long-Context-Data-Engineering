"""
Code adapted from InfiniBench: https://github.com/OpenBMB/InfiniteBench/tree/main
The original author has the copy right.

# ours 7B longbook qa test to 128k
(
python -u eval_book.py --task longbook_qa_eng\
    --verbose\
    --model_path ../../../llama-2-7b-80k\
    --data_dir ../../../InfiniteBench/data\
    --model_name llama\
    --truncate 128000
) 2>&1  | tee logs/eval_llama_7b_80k_test_to_128k.log

# ours 13B longbook qa, test to 128k
(
python -u eval_yarn_mistral.py --task longbook_qa_eng\
    --verbose\
    --model_path ../../../llama-2-13b-64k\
    --data_dir ../../../InfiniteBench/data\
    --model_name llama\
    --truncate_len 128000
) 2>&1  | tee logs/eval_3.1.2.0_s1000_test_to_128k.log
"""
import numpy as np 
import json
from pathlib import Path
import time
import math
from typing import List, Tuple, Any
import tensor_parallel as tp

import torch
from torch import Tensor
from transformers import AutoTokenizer
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers import AutoModelForCausalLM, AutoTokenizer,AutoConfig
import transformers
from eval_utils import (
    dump_jsonl,
    create_prompt,
    load_data,
    get_answer,
    DATA_NAME_TO_MAX_NEW_TOKENS,
    process_choice
)
# from yarn_mistral.modeling_mistral_yarn import MistralForCausalLM
from args import parse_args

# from yarn_load import load_model_and_apply_patches
# from llama_config_monkey_patch import replace_llama_config
# replace_llama_config()


MAX_POSITION_ID = 128 * 1024  # Determined by the model
TRUNCATE_LEN = 30000

# from llama_attn_replace import replace_llama_attn

def reset_rope(model, model_max_train_len, scaling_factor):
    for l in model.model.layers:
        l.self_attn.rotary_emb.scaling_factor = scaling_factor
        l.self_attn.rotary_emb._set_cos_sin_cache(seq_len=model_max_train_len, device="cpu", dtype=torch.float32)
    return


def truncate_input(input: list, max_length: int, manner="middle"):
    if len(input) <= max_length:
        return input
    if manner == "middle":
        split = max_length // 2
        return input[0:split] + input[-split:]
    else:
        return None


def truncate_by_tokens(input, tok, max_tokens, manner: str = "middle"):
    tokens = tok.encode(input)
    len_before = len(tokens)
    print(f"# tokens before: {len_before}")
    tokens = truncate_input(tokens, max_length=max_tokens, manner=manner)
    len_after = len(tokens)  # type: ignore
    print(f"# tokens after: {len_after}")
    assert len_after <= len_before
    assert len_after <= max_tokens
    return tok.decode(tokens, skip_special_tokens=True)


def chunk_generate(
    model,
    tok,
    texts: List[str],
    max_tokens: int,
    sliding_window: int = 128 * 1024,
    chunk_size: int = 2500,
    verbose: bool = False,
    chunked: bool = False,
) -> List[str]:
    """
    Directly performing inference using HF transformers will result in OOM
    when using one A100 GPU. This is because the attention matrix is too large,
    so we chunk the input up and perform forward pass on each chunk to build
    up the KV cache. Note that each token still has to attend to
    all tokens in the past.
    """
    with torch.no_grad():
        """
        input_ids: (b, n)
        attention_mask: (b, n)
        [
            [0, 0, .., 0, 1, 1, ..., 1]
            ...
        ]
        """
        inputs = tok(texts, return_tensors="pt", padding=True)
        inputs = inputs.to(model.device)  # type: ignore
        input_ids: Tensor = inputs.input_ids  # (b, n)
        attention_mask: Tensor = inputs.attention_mask  # (b, n)
        position_ids: Tensor = attention_mask.long().cumsum(dim=-1) - 1
        position_ids.masked_fill_(attention_mask == 0, value=1)
        seq_len = input_ids.shape[-1]
        print("seq_len:", seq_len)
        kv_cache: Any = None
        # Split into chunks for pre-filling
        chunk_idxs = []
        n = seq_len - 1
        while n > 0:
            chunk_idxs.append(n)
            n -= chunk_size
        chunk_idxs.append(0)
        chunk_idxs = chunk_idxs[::-1]
        chunk_lo = chunk_idxs[:-1]
        chunk_hi = chunk_idxs[1:]
        print(f"Number of chunks: {len(chunk_lo)}, generating...")
        start_time = time.time()
        if(chunked):
            for chunk_i, (chunk_lo, chunk_hi) in enumerate(
                zip(chunk_lo, chunk_hi)
            ):
                if verbose:
                    print(
                        f"[chunk {chunk_i}] {chunk_lo} : {chunk_hi}",
                        round(time.time() - start_time),
                    )
                chunk_input_ids = input_ids[:, chunk_lo:chunk_hi]
                if kv_cache is not None:
                    mask_start_idx = chunk_lo - kv_cache[0][0].shape[2]
                else:
                    mask_start_idx = chunk_lo
                chunk_attention_mask = attention_mask[:, mask_start_idx:chunk_hi]
                chunk_position_ids = position_ids[:, chunk_lo:chunk_hi]
                outputs: BaseModelOutputWithPast = model.model.forward(
                    input_ids=chunk_input_ids,
                    attention_mask=chunk_attention_mask,
                    position_ids=chunk_position_ids,
                    past_key_values=kv_cache,
                    return_dict=True,
                    use_cache=True,
                )
                kv_cache = outputs.past_key_values
                # Discard KV states on the left beyond the window
                new_cache = ()
                n_layers = len(kv_cache)
                for layer_i in range(n_layers):
                    keys = kv_cache[layer_i][0][:, :, -sliding_window:]
                    values = kv_cache[layer_i][1][:, :, -sliding_window:]
                    new_cache += ((keys, values),)
                kv_cache = new_cache
            kv_cache_len = kv_cache[0][0].shape[2]
            outputs = model.generate(
                input_ids=input_ids[:, -1:],
                attention_mask=attention_mask[:, -kv_cache_len - 1 :],
                max_new_tokens=max_tokens,
                past_key_values=kv_cache,
                eos_token_id=tok.pad_token_id,
                use_cache=True,
            )
            responses = [
                tok.decode(t[1:], skip_special_tokens=True) for t in outputs
            ]
        else:
            outputs = model.generate(
                input_ids=input_ids[:, -1:],
                max_new_tokens=max_tokens,
                eos_token_id=tok.pad_token_id,
            )
            responses = [
                tok.decode(t[1:], skip_special_tokens=True) for t in outputs
            ]
    return responses


def get_pred(
    model,
    tok: AutoTokenizer,
    input_text: str,
    max_tokens: int,
    verbose: bool = False,
    truncate_len: int = 80000,
) -> str:
    """
    Truncate down to 128k then make inference.
    """
    print("Truncating...")
    input_text = truncate_by_tokens(input_text, tok, truncate_len)
    inputs = tok(input_text, return_tensors="pt")
    inputs = inputs.to(model.device)  # type: ignore
    input_ids: Tensor = inputs.input_ids  # (b, n)
    attention_mask: Tensor = inputs.attention_mask
    #print(input_ids[:,:10], tok.decode(input_ids[0,:10], skip_special_tokens=False))
    outputs = model.generate(input_ids=input_ids, max_new_tokens=max_tokens) 
    output = tok.decode(outputs[0][input_ids.size(1):],skip_special_tokens=True)
    if verbose:
        print("# chars:", len(input_text))
        print("=============== Input ===============")
        print(input_text[:200])
        print("...")
        print(input_text[-200:])
        print("=====================================")
    
    # print(output)
    return output


def load_model(
    args,
    model_name: str = "../../../yarn-mistral-7b-128k",
    ):
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    tok.pad_token = tok.eos_token
    start_time = time.time()
    if("yarn" in model_name.lower()):
        # NOTE: need to follow YaRN setup from https://github.com/jquesnelle/yarn
        # ATTENTION: you cannot directly load the model from HF because YaRN's author modifies the HF implementation
        # so make sure you follow the official YaRN setup
        raise NotImplementedError
        model = load_model_and_apply_patches(model_name, args)
    elif("longlora" in model_name): 
        # NOTE: need to follow longlora setup https://github.com/dvlab-research/LongLoRA
        raise NotImplementedError
        replace_llama_attn(inference=True)
        config = transformers.AutoConfig.from_pretrained(
                model_name
            )
        scaling_factor = float(math.ceil(args.truncate_len / 4096))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}
        model = AutoModelForCausalLM.from_pretrained(
                model_name,
                config=config,
                torch_dtype=torch.float16,
            ).eval()
        model.resize_token_embeddings(32001)
    elif("together" in model_name):
        # NOTE: this one you can directly load from HF without change modification
        model = AutoModelForCausalLM.from_pretrained(model_name, use_flash_attention_2="flash_attention_2", 
                                                     torch_dtype=torch.bfloat16,
                                                     ).eval()
        scaling_factor = 32 # hardcode
        reset_rope(model, model_max_train_len=81920, scaling_factor=scaling_factor)
    else: 
        # Our LLaMA-2 7B 80K
        model = AutoModelForCausalLM.from_pretrained(model_name, use_flash_attention_2="flash_attention_2", 
                                                     torch_dtype=torch.bfloat16,
                                                     ).eval()
        scaling_factor = 10 # hardcode
        reset_rope(model, model_max_train_len=81920, scaling_factor=scaling_factor)
    model = tp.tensor_parallel(model, sharded=True)
    return model, tok


if __name__ == "__main__":
    
    args = parse_args()
    model_name = args.model_name
    print(json.dumps(vars(args), indent=4))
    data_name = args.task

    # Model
    max_tokens = DATA_NAME_TO_MAX_NEW_TOKENS[data_name]
    model, tok = load_model(args, args.model_path)
    
    # Data
    result_dir = Path(args.output_dir, model_name)
    result_dir.mkdir(exist_ok=True, parents=True)
    examples = load_data(data_name, data_dir=args.data_dir)

    if args.stop_idx is None:
        args.stop_idx = len(examples)
        output_path = (
            result_dir / f"preds_{data_name}.jsonl"
        )
    else:
        output_path = (
            result_dir / f"preds_{data_name}_{args.start_idx}-{args.stop_idx}.jsonl"  # noqa
        )

    preds = []
    print("==== Evaluation ====")
    print(f"# examples: {len(examples)}")
    print(f"Start index: {args.start_idx}")
    print(f"Stop index: {args.stop_idx}")
    print(f"Verbose: {args.verbose}")
    print(f"Max tokens: {max_tokens}")
    start_time = time.time()
    recall = []
    for i in range(args.start_idx, args.stop_idx):
        eg = examples[i]
        input_text = create_prompt(eg, data_name, "llama", args.data_dir)
        # import ipdb; ipdb.set_trace()
        time_so_far = time.time() - start_time
        minute = int(time_so_far / 60)
        second = int(time_so_far % 60)
        print(f"====== Example %d / %d Time %d m %d s ======" % (i, len(examples), minute, second))
        pred = get_pred(
            model, tok, input_text, max_tokens=max_tokens, verbose=args.verbose, truncate_len=args.truncate_len
        )
        # if args.verbose:
        #     print(pred)
        preds.append(
            {
                "id": i,
                "prediction": pred,
                "ground_truth": get_answer(eg, data_name),
            }
        )
        # print("Ground truth:", get_answer(eg, data_name))
        pred = pred.lower()
        if("choice" in data_name):
            pred = process_choice(pred, eg["options"])
        pred = pred.split()
        print("pred:")
        print(" ".join(pred))
        ground_truth = get_answer(eg, data_name)[0].lower().split()
        print("ground truth:")
        print(" ".join(ground_truth))
        recall_ = len(set(ground_truth).intersection(set(pred))) / len(set(ground_truth))
        recall.append(recall_)
        print("Recall so far %.3f" % np.average(recall))
        dump_jsonl(preds, output_path)
