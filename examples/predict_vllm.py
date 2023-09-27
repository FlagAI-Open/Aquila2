from vllm import LLM, SamplingParams
import torch
import time
import random
import numpy as np
import math

prompts = ['请给出10个要到北京旅游的理由。',]
sampling_params = SamplingParams(temperature=0.9, top_p=0.95,top_k=10000, max_tokens=50, stop="</s>", logprobs=1, get_prompt_logprobs=True)

llm = LLM(model="/share/xw/aquila_30B/iter_0082000_hf", tensor_parallel_size=2, trust_remote_code=True, dtype='bfloat16')

from transformers import (AutoTokenizer, PreTrainedTokenizer,
                          PreTrainedTokenizerFast)
def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(
            './checkpoint/aquila2_7B',
            trust_remote_code=True)
    return tokenizer

#llm = LLM(model="./checkpoint/aquila2_7B", trust_remote_code=True, tensor_parallel_size=1, dtype='bfloat16')
tokenizer = get_tokenizer()
vocab = tokenizer.vocab

id2word = {v:k for k, v in vocab.items()}

start_time = time.time()
outputs = llm.generate(prompts, sampling_params, stream=True)
end_time = time.time()
# Print the outputs.
for output in outputs:
    res_dic = dict()
    for prompt, logprob in zip(output.prompt_token_ids, output.outputs[0].logprobs):
        res_dic[prompt] = math.exp(logprob[prompt]) if logprob is not None else 0
    print(output.outputs[0].text)

execution_time = end_time - start_time
print(f"执行时间：{execution_time}秒")
current_memory = torch.cuda.max_memory_allocated() / 1024 ** 3
print(f"当前内存使用量：{current_memory} GB")
