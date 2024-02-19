import datetime
from typing import Any
import jsonlines
from time import time
from multiprocessing import Pool 
from datasets import Dataset, concatenate_datasets

LLAMA_CHAR_TO_TOKEN_RATIO = 3.68 # estimated from tests/test_llama_char_to_token_ratio.py

SLIMPAJAMA_ORIGINAL_MIX = {"RedPajamaC4": 0.26609933,
                           "RedPajamaCommonCrawl": 0.53374772,
                           "RedPajamaStackExchange": 0.03169647,
                           "RedPajamaWikipedia": 0.03402128,
                           "RedPajamaGithub": 0.05004782,
                           "RedPajamaArXiv": 0.04290379, 
                           "RedPajamaBook": 0.0414836
                           }

UPSAMPLE_CODE_ARXIV_BOOK = {"RedPajamaC4": 0.26609933,
                           "RedPajamaCommonCrawl": 0.53374772,
                           "RedPajamaStackExchange": 0.03169647,
                           "RedPajamaWikipedia": 0.03402128,
                           "RedPajamaGithub": 0.60,
                           "RedPajamaArXiv": 0.60, 
                           "RedPajamaBook": 0.60
                           }

MEDIUM_LENGTH = {"RedPajamaC4": 315,
                 "RedPajamaCommonCrawl": 1066,
                 "RedPajamaStackExchange": 507,
                 "RedPajamaWikipedia": 409,
                 "RedPajamaGithub": 520,
                 "RedPajamaArXiv": 14565, 
                 "RedPajamaBook": 121602,
                }

def tprint(msg="", **kwargs):
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if(msg == ""): print()
    else: print(f"[{current_time}] {msg}", **kwargs, flush=True)

def convert_time(s):
    """Convert seconds to days, hours, minutes"""
    days = s // (60 * 60 * 24)
    hours = s // (60 * 60) % 24
    minutes = s // 60 % 60
    return (days, hours, minutes)

def estimate_time(start_time, processed_tokens, total_tokens, return_sec=False):
    """Estimate the time used, time remaining, and total time"""
    time_so_far = time() - start_time
    time_so_far_formated = convert_time(time_so_far)
    total_esimate = time_so_far / processed_tokens * total_tokens
    total_esimate_formated = convert_time(total_esimate)
    remining = total_esimate - time_so_far
    remining_formated = convert_time(remining)
    if(return_sec):
        return time_so_far_formated, total_esimate_formated, remining_formated, time_so_far
    else:
        return time_so_far_formated, total_esimate_formated, remining_formated

def read_jsonl(path, is_print=True):
    """Read the jsonl file
    NOTE: huggingface load_dataset function is much faster than this function. Do not know why
    """
    data = []
    i = 0
    with jsonlines.open(path) as f:
        for l in f: 
            data.append(l)
            if(i % 10000 == 0 and is_print): tprint("%d lines read" % i)
            i += 1
    return data

def load_convert(p):
    data = read_jsonl(p, is_print=False)
    data = Dataset.from_list(data)
    return data

def load_dataset(path, n_shard):
    """Load a pre-sharded dataset"""
    tprint("Loading from %s" % path)
    start_time = time()
    with Pool(n_shard) as pool:
        paths = [path + "_%d-%d.jsonl" %(i, n_shard) for i in range(n_shard)]
        data = pool.map(load_convert, paths)
        data = concatenate_datasets(data)
    print("%d seconds in total" % (time() - start_time))
    return data

class TimedTokenizer(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.used_time = 0
        return 

    def __call__(self, *args, **kwargs):
        start_time = time()
        ret = self.tokenizer(*args, **kwargs)
        self.used_time += time() - start_time
        return ret