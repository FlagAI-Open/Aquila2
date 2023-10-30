# Initially adopted from https://github.com/ouwei2013/aquila2_34b_awq.git

# Usage: 
## 1. download AquilaChat2-34B-AWQ files from https://model.baai.ac.cn/model-detail/100122
## 2. install AutoAWQ==v0.1.5 from https://github.com/casper-hansen/AutoAWQ

import torch

from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

awq_model_path = './checkpoints/aquilachat2-34b-awq'
model = AutoAWQForCausalLM.from_quantized(awq_model_path,trust_remote_code=True,fuse_layers=True)
tokenizer = AutoTokenizer.from_pretrained(awq_model_path,trust_remote_code=True)
model.eval()

device = torch.device("cuda:0")
model.to(device)

text = "请给出10个要到北京旅游的理由。"
from flagai.model.aquila2.utils import covert_prompt_to_input_ids_with_history
history = None
text = covert_prompt_to_input_ids_with_history(text, history, tokenizer, 2048, convo_template="aquila-legacy")
inputs = torch.tensor([text]).to(device)
outputs = model.generate(inputs)[0]
print(tokenizer.decode(outputs))
