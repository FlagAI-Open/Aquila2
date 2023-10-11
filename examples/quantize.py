import sys;sys.path.append("/data2/yzd/git/AutoGPTQ")
from transformers import AutoTokenizer, TextGenerationPipeline
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
import logging
import torch
from flagai.auto_model.auto_loader import AutoLoader
from transformers import BitsAndBytesConfig


# logging.basicConfig(
#     format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
# )


model_name = 'AquilaChat-7B'

quantize_config = BaseQuantizeConfig(
    bits=4,  # quantize model to 4-bit
    group_size=128,  # it is recommended to set the value to 128
    desc_act=False,  # set to False can significantly speed up inference but the perplexity may slightly bad
)

model = AutoGPTQForCausalLM.from_pretrained('./checkpoints/'+model_name, quantize_config, trust_remote_code=True,)
tokenizer = AutoTokenizer.from_pretrained('./checkpoints/'+model_name, use_fast=True,trust_remote_code=True)
examples = [
    tokenizer(
        "auto-gptq is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm."
    )
]
# pretrained_model_dir = "facebook/opt-125m"
# quantized_model_dir = "opt-125m-4bit"

# tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir, use_fast=True)
# examples = [
#     tokenizer(
#         "auto-gptq is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm."
#     )
# ]



# # load un-quantized model, by default, the model will always be loaded into CPU memory
# model = AutoGPTQForCausalLM.from_pretrained(pretrained_model_dir, quantize_config)

# quantize model, the examples should be list of dict whose keys can only be "input_ids" and "attention_mask"
import pdb;pdb.set_trace()
quantize_model_dir = 'test_checkpoints'
model.quantize(examples)

# save quantized model
model.save_quantized(quantized_model_dir)