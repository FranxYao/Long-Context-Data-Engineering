"""
Simply packing the slim pajama dataset. Distributed version
NOTE: may need to run `huggingface-cli login` first

```bash
# Generate a file of 11B tokens, per source filtering 
# takes 1 hour to finish
PATH_TO_SLIMPAJAMA=YOUR_PATH_HERE
nohup python -u slimpajama_packing.py --dataset_size=11b --print_interval=100 --num_process=200\
    --dataset_path=$PATH_TO_SLIMPAJAMA\
    --output_path=../data/slimpajama/per_source_downsample/ --down_sample_ratio=0.1 --down_sample_mode=per_source\
    > ../logs/slimpajama_packing_dist_per_source_downsample_0.1.log 2>&1 &
tail -f ../logs/slimpajama_packing_dist_per_source_downsample_0.1.log
"""

import jsonlines
import datasets
import transformers
import argparse
import pickle 
import random 

import numpy as np

# from fim import make_fim, update_fim_stats, report_fim_stats
from time import time, sleep
from datasets import load_dataset
from transformers import LlamaTokenizer
from multiprocessing import Process, Pipe, Lock, Event
from utils import tprint, estimate_time, TimedTokenizer, SLIMPAJAMA_ORIGINAL_MIX, LLAMA_CHAR_TO_TOKEN_RATIO, UPSAMPLE_CODE_ARXIV_BOOK

np.random.seed(15213)

def get_args():
    parser = argparse.ArgumentParser()
    
    # NOTE: currently only support mixing llama data and Python portion of starcoder data 
    parser.add_argument('--dataset_size', type=str, default="5b")
    parser.add_argument('--dataset_path', type=str, default="../data/SlimPajama-627B/")
    parser.add_argument('--output_path', type=str, default="../data/slimpajama/")
    parser.add_argument('--chunk_size', type=int, default=131072)
    parser.add_argument('--print_interval', type=int, default=10)
    parser.add_argument('--num_process', type=int, default=1)
    parser.add_argument('--down_sample_mode', type=str, default="none", choices=["none", "global", "per_source", "upsample_code_arxiv_book"])
    parser.add_argument('--down_sample_ratio', type=float, default="0.2", help="down sample to this ratio")
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--fim_ratio', type=float, default=0., help="ratio for creating fill-in-the-middle data")
    
    args = parser.parse_args()
    return args

def report_length_dist(length_dist_by_source):
    total_tokens = sum(sum(length_dist_by_source[source]) for source in length_dist_by_source)
    for source in length_dist_by_source:
        length_dist = np.array(length_dist_by_source[source])
        percentiles = [25, 50, 75, 90, 95, 99, 100]
        results = np.percentile(length_dist, percentiles)
        tprint("    data source %s portion %.3f " % 
            (source, sum(length_dist_by_source[source]) / total_tokens), end="")
        for p, r in zip(percentiles, results):
            print("    %d : %d, " % (p, r), end="")
        print()
    return 

def update_source_loc_index(buffer_source, buffer_size):
    """split buffer_source structure that record the source of data within a buffer 
    
    Example:
    Input:
        buffer_source = [
                {"source": "arxiv", "start": 0, "end": 50},
                {"source": "cc", "start": 50, "end": 80},
                {"source": "c4", "start": 80, "end": 100},
            ]
        buffer_size = 60

    Output:
        new_loc_index = [
                {"source": "arxiv", "start": 0, "end": 50},
                {"source": "cc", "start": 50, "end": 60},
            ]
        splitted_loc_index = [
                {"source": "cc", "start": 0, "end": 20},
                {"source": "c4", "start": 20, "end": 40},
            ]
    """
    new_loc_index = []
    splitted_loc_index = []
    for loc in buffer_source:
        if(loc["end"] <= buffer_size):
            new_loc_index.append(loc)
        elif(loc["start"] < buffer_size):
            new_loc_index.append({
                "source": loc["source"],
                "start": loc["start"],
                "end": buffer_size,
            })
            splitted_loc_index.append({
                "source": loc["source"],
                "start": 0, 
                "end": loc["end"] - buffer_size,
            })
        else:
            splitted_loc_index.append({
                "source": loc["source"],
                "start": loc["start"] - buffer_size,
                "end": loc["end"] - buffer_size,
            })
    return new_loc_index, splitted_loc_index

def tokenizer_process(tokenizer, conn_recv, conn_writer, lock, rank, is_available):
    while(True):
        d = conn_recv.recv()
        # print("rank %d tokenizing" % rank)
        if(isinstance(d, str) and d == "FINISH"):
            break
        tokenized = tokenizer(d["text"])

        lock.acquire()
        conn_writer.send((tokenized, d))
        lock.release()

        is_available.set() # notice the main process that this tokenizer is available
    return 

def writer_process(f, args, conn, num_samples):
    """Process for writing the data to the disk"""
    buffer = []
    buffer_source = []
    buffer_size = 0
    per_source_length_dist_after_down_sample = {}
    loop_end = False

    idx = 0
    start_time = time()
    while(True):
        tokenized, d = conn.recv()
        # tokenized = tokenizer(d["text"])
        if(isinstance(tokenized, str) and tokenized == "FINISH"):
            break

        dlen = len(tokenized["input_ids"])
        source = d["meta"]["redpajama_set_name"]
        start_loc = buffer_size
        buffer_size += dlen
        end_loc = buffer_size
        buffer.extend(tokenized["input_ids"])
        buffer_source.append(
            {"source": source, "start": start_loc, "end": end_loc})

        while buffer_size >= args.chunk_size + 1:
            # add data to chunk
            data = {}
            data["input_ids"] = buffer[:args.chunk_size]    # input always by default 16K
            data["labels"] = buffer[args.chunk_size]
            data_source, buffer_source = update_source_loc_index(buffer_source, args.chunk_size)
            data["source"] = data_source

            for s_ in data_source:
                source = s_["source"]
                dlen = s_["end"] - s_["start"]
                if(source in per_source_length_dist_after_down_sample): 
                    per_source_length_dist_after_down_sample[source].append(dlen)
                else: per_source_length_dist_after_down_sample[source] = [dlen]
            
            f.write(data)
            idx += 1
            if(idx % args.print_interval == 0): 
                time_so_far, total_esimate, remining, time_so_far_in_sec = estimate_time(start_time, idx, num_samples, True)
                tprint("global process %d / %d = %.3g, time used %d:%d:%d + estimated remain %d:%d:%d = total %d:%d:%d" 
                        % (idx, num_samples, idx / num_samples, *time_so_far, *remining, *total_esimate))
                
            buffer = buffer[args.chunk_size : ]
            buffer_size -= args.chunk_size

    tprint("Per source length distribution, after downsample:")
    report_length_dist(per_source_length_dist_after_down_sample)

    pickle.dump(per_source_length_dist_after_down_sample,
        open(args.output_path + "per_source_length_dist_after_%s_down_sample_%.3g.pkl" % 
             (args.down_sample_mode, args.down_sample_ratio), "wb"), 
        )
    print("Finished")
    return 


def main():
    args = get_args()

    ## compute the total number of tokens
    if(args.dataset_size == "test"):
        total_tokens = 2000000
    elif(args.dataset_size[-1] == "b"):
        B = int(1e9)
        total_tokens = int(args.dataset_size[:-1]) * B
    elif(args.dataset_size[-1] == "m"):
        M = int(1e6)
        total_tokens = int(args.dataset_size[:-1]) * M
    else: raise ValueError("dataset size %s not supported" % args.dataset_size)

    num_samples = total_tokens // args.chunk_size

    tprint("setting total tokens %d" % total_tokens)

    ## load the slimpajama stream dataloader and tokenizer
    dataset = load_dataset(args.dataset_path, streaming=True, split="train")
    tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    tokenizer = TimedTokenizer(tokenizer)

    ## loop over the stream dataloader, packing and record the data source
    # Note that the shuffle algorithm of dataset is nontrivial, see
    # https://huggingface.co/docs/datasets/v2.15.0/en/package_reference/main_classes#datasets.IterableDataset.shuffle
    if(args.debug == False):
        dataset = dataset.shuffle(42)

    if(args.down_sample_mode == "per_source"):
        per_source_budget = {s: total_tokens * SLIMPAJAMA_ORIGINAL_MIX[s] for s in SLIMPAJAMA_ORIGINAL_MIX}
        per_source_tokens = {s: 0 for s in SLIMPAJAMA_ORIGINAL_MIX}
        per_source_filtered = {s: {"filtered": 0, "passed": 0} for s in SLIMPAJAMA_ORIGINAL_MIX}
    
    if(args.down_sample_mode == "upsample_code_arxiv_book"):
        mix_ratio_normalizer = sum(UPSAMPLE_CODE_ARXIV_BOOK.values())
        mix_ratio = {s: UPSAMPLE_CODE_ARXIV_BOOK[s] / mix_ratio_normalizer for s in UPSAMPLE_CODE_ARXIV_BOOK}
        per_source_budget = {s: total_tokens * mix_ratio[s] for s in mix_ratio}
        per_source_tokens = {s: 0 for s in SLIMPAJAMA_ORIGINAL_MIX}
        per_source_filtered = {s: {"filtered": 0, "passed": 0} for s in SLIMPAJAMA_ORIGINAL_MIX}

    if(args.down_sample_mode == "none"):
        if(args.fim_ratio > 0):
            data_file_name = args.output_path + "slimpajama_packed_%d_%s_fim_%.2f.jsonl" %\
                            (args.chunk_size, args.dataset_size, args.fim_ratio)
        else:
            data_file_name = args.output_path + "slimpajama_packed_%d_%s.jsonl" % (args.chunk_size, args.dataset_size)
    else: 
        if(args.fim_ratio > 0):
            data_file_name = args.output_path + "slimpajama_packed_%d_%s_%s_down_sample_%.3g_fim_%.2f.jsonl" %\
                            (args.chunk_size, args.dataset_size, args.down_sample_mode, args.down_sample_ratio, args.fim_ratio)
        else:
            data_file_name = args.output_path + "slimpajama_packed_%d_%s_%s_down_sample_%.3g.jsonl" %\
                                (args.chunk_size, args.dataset_size, args.down_sample_mode, args.down_sample_ratio)
    tprint("Output to %s" % data_file_name)
    f = jsonlines.open(data_file_name, mode="w")

    # tokenizer process
    pipe_to_writer, writer_conn = Pipe()
    lock = Lock()
    tokenizer_pipes = []
    processes = []
    tokenizer_availability = []
    for rank in range(args.num_process):
        main_conn, tokenizer_conn = Pipe()
        is_available = Event()
        is_available.set()
        tokenizer_availability.append(is_available)
        tokenizer_pipes.append(main_conn)
        p = Process(target=tokenizer_process, args=(tokenizer, tokenizer_conn, pipe_to_writer, lock, rank, is_available))
        processes.append(p)
        p.start()

    # writer process
    p_writer = Process(target=writer_process, 
                args=(f, args, writer_conn, num_samples))
    p_writer.start()

    def _find_available_tokenizer():
        # find an available tokenizer
        found = False
        while(found == False):
            for rank, is_available in enumerate(tokenizer_availability):
                if(is_available.is_set()):
                    is_available.clear()
                    found = True
                    tokenizer_pipe = tokenizer_pipes[rank]
                    break
            if(found == False): sleep(0.1)
        return tokenizer_pipe
        
    # main process doing the filtering, then send to tokenizer
    start_time = time()
    # tokenizer_pid = 0
    processed_token = 0
    di = 0
    per_source_length_dist = {}
    for d in dataset:
        # filter out short doc using global or per-source filtering
        dlen = len(d["text"]) / LLAMA_CHAR_TO_TOKEN_RATIO
        source = d["meta"]["redpajama_set_name"]
        if(source in per_source_length_dist): per_source_length_dist[source].append(dlen)
        else: per_source_length_dist[source] = [dlen]

        # downsample at the medium length 
        if(args.down_sample_mode == "global"):
            # global downsample, this method CHANGEs mixture ratio, but not much
            if(dlen < 4096 and np.random.random() > args.down_sample_ratio): continue
        elif(args.down_sample_mode in ["per_source", "upsample_code_arxiv_book"]): 
            # the budget of the corresponding source is used up, do not add new data
            if(per_source_tokens[source] > per_source_budget[source]): continue
            else:
                # short sequence, filter out
                if(dlen < 4096 and np.random.random() > args.down_sample_ratio): 
                    per_source_filtered[source]["filtered"] += 1
                    continue
                else: # not short, add and record number of tokens
                    per_source_filtered[source]["passed"] += 1
                    per_source_tokens[source] += dlen
        else: 
            pass # no downsample
        di += 1
        
        # fim_stats = {"total_case": 0, "fim_case": 0, "prefix_len": [], "middle_len": [], "suffix_len": []}
        # if(random.random() < args.fim_ratio):            
        #     d = make_fim(d, content_key="text")
        #     update_fim_stats(fim_stats, d)

        tokenizer_pipe = _find_available_tokenizer()
        tokenizer_pipe.send(d)
        # tokenizer_pipes[tokenizer_pid].send(d)
        # tokenizer_pid = (tokenizer_pid + 1) % args.num_process
        processed_token += dlen

        # report progress
        if(di % 1000 * args.print_interval == 0 and args.down_sample_mode in ["per_source", "upsample_code_arxiv_book"]):
            tprint("%dth instance" % di)
            for s in per_source_tokens:
                tprint("%s %d / %d = %.2g, filtered %d passed %d" % 
                    (s, per_source_tokens[s], per_source_budget[s], per_source_tokens[s] / per_source_budget[s],
                    per_source_filtered[s]["filtered"], per_source_filtered[s]["passed"]))

        # finish condition
        if(args.down_sample_mode in ["global", "none"]):
            if(processed_token > total_tokens): break
        elif(args.down_sample_mode in ["per_source", "upsample_code_arxiv_book"]):
            is_finished = True
            for source in per_source_tokens:
                if(per_source_tokens[source] < per_source_budget[source]): 
                    is_finished = False
                    break
            if(is_finished): break
        else: raise NotImplementedError
    
    for conn in tokenizer_pipes: conn.send(("FINISH")) # tell tokenizer process to finish up
    for p in processes: p.join() # wait for tokenizer process finishing up

    tprint("Per source length distribution, original:")
    report_length_dist(per_source_length_dist)
    pipe_to_writer.send(("FINISH", None))
    p_writer.join()
    return 

if __name__ == "__main__":
    main()