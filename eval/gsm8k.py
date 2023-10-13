''' Copied from https://github.com/QwenLM/Qwen/tree/main/eval. '''

import re
import os
import json
import torch
import argparse
import jsonlines
import numpy as np
import datasets
from tqdm import tqdm
from datasets import load_from_disk, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"

def doc_to_text(doc):

    return (
        fewshot_prompt
        + "\nQuestion: "
        + doc["question"]
        + "\nLet's think step by step\n"
    )

def decode(tokens_list, tokenizer, raw_text_len):
    sents = []
    # print(len(tokens_list))
    for tokens in tokens_list:
        tokens = tokens.cpu().numpy().tolist()
        sent = tokenizer.decode(tokens[raw_text_len:])
        sent = sent.split("<|endoftext|>")[0]
        sent = sent.split("\n\n\n")[0]
        sent = sent.split("\n\n")[0]
        sent = sent.split("Question:")[0]
        sents.append(sent)
    return sents

def generate_sample(model, tokenizer, input_txt):
    input_ids = tokenizer.encode(input_txt)
    raw_text_len = len(input_ids)
    context_enc = torch.tensor([input_ids]).to(model.device)
    # print(f"Input text: {input_txt}\n")
    outputs = model.generate(context_enc, do_sample=False, max_length=2048, eos_token_id=100007)
    output_text = decode(outputs, tokenizer, raw_text_len)[0]
    # print(f"\nOutput text: {output_text}\n")
    return output_text

def extract_answer_hf(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return eval(match_str)
    else:
        return INVALID_ANS


def extract_answer(completion):
    try:
        last_number = re.findall(r"\d+", completion)[-1]
        return eval(last_number)
    except:
        return INVALID_ANS

def is_correct(completion, answer):
    gold = extract_answer_hf(answer)
    assert gold != INVALID_ANS, "No ground truth answer found in the document."
    return extract_answer(completion) == gold

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test HF checkpoint.")
    parser.add_argument(
        "-c",
        "--model",
        type=str,
        help="Checkpoint path",
        default="./checkpoints-in",
    )
    parser.add_argument(
        "-o", "--sample-output-file", type=str, default="gsm8k_res_1999-hf_fewshot-pt.jsonl"
    )
    parser.add_argument("-d", "--device", type=str, default='1')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device

    fewshot_prompt = open("gsm8k_prompt.txt").read()
    data = []
    if args.sample_input_file is not None:
        with open(args.sample_input_file) as f:
            for item in f.readlines():
                data.append(json.loads(item))
   

    test = data

    print("Loading tokenizer ...")
 

    print("Loading model ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16) 
    #ckpt = torch.load(args.checkpoint_path, map_location='cpu')
    #model.load_state_dict(ckpt, strict=False)
    model.eval()
    model.to("cuda:0")

    f_output = jsonlines.Writer(open(args.sample_output_file, "w", encoding="utf-8"))
    # tot_length = test.num_rows
    acc_res = []
    for doc in tqdm(test):
        context = doc_to_text(doc)
        completion = generate_sample(model, tokenizer, context)
        answer = doc["answer"]

        acc = is_correct(completion, answer)
        doc["completion"] = completion
        print(completion)
        doc["acc"] = acc
        f_output.write(doc)
        acc_res.append(acc)

    f_output.close()
    print("Acc: ", np.mean(acc_res))
