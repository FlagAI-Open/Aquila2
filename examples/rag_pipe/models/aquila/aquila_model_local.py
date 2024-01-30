# !/usr/bin/env python
# -*- coding:utf-8 -*-
# ==================================================================
# [CreatedDate]  : Tuesday, 2023-07-04 09:19:30
# [Author]       : shixiaofeng
# [Descriptions] :
# ==================================================================
# [ChangeLog]:
# [Date]    	[Author]	[Comfments]
# ------------------------------------------------------------------
# https://huggingface.co/docs/transformers/v4.34.1/en/internal/generation_utils#transformers.TextIteratorStreamer
import time

import torch

# 建议不安装，直接clone代码库，再将路径加入pythonpath中
# import sys
# sys.path.insert(0, "/share/project/shixiaofeng/code/FlagAI-official")

from flagai.model.aquila2.modeling_aquila import AquilaForCausalLM
from flagai.model.aquila2.utils import covert_prompt_to_input_ids_with_history

from threading import Thread

from accelerate import load_checkpoint_and_dispatch
from transformers import AutoTokenizer, TextIteratorStreamer, GenerationConfig
import copy


class AquliaModel(object):
    def __init__(self, model_info):
        self.setup_model(model_info)

    def setup_model(self, model_info):
        model_dir = model_info["local_model_path"]
        model_name = model_info["name"]
        qlora_dir = model_info.get("qlora_dir", "")
        self.device = "cuda"

        start_time = time.time()
        print(f"AquliaModel model_dir: [{model_dir}]")
        print(f"AquliaModel model_name: [{model_name}]")

        low_cpu_mem_usage = True
        quantize = False
        torch_dtype = torch.float16

        if "34b" in model_name.lower():
            torch_dtype = torch.bfloat16

        quantization_config = None
        print(f"quantize is {quantize}")
        if quantize:
            from transformers import BitsAndBytesConfig

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch_dtype,
            )

        model = AquilaForCausalLM.from_pretrained(
            model_dir,
            low_cpu_mem_usage=low_cpu_mem_usage,
            torch_dtype=torch_dtype,
            quantization_config=quantization_config,
        )

        model.eval()

        if not quantization_config:
            model = load_checkpoint_and_dispatch(
                model,
                model_dir,
                device_map="balanced",
                no_split_module_classes=["LlamaDecoderLayer", "AquilaDecoderLayer"],
            )

        if qlora_dir:
            from flagai.model.tools.peft import PeftModel

            model = PeftModel.from_pretrained(model, qlora_dir)
            print("Load Qlora Adaptor")

        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        end_time = time.time()
        print(f"Load model timecost is: {end_time - start_time}s")

        self.generation_cfg = {
            "bos_token_id": 100006,
            "eos_token_id": 100007,
            "pad_token_id": 0,
            "use_cache": True,
        }
        self.model = model
        self.tokenizer = tokenizer
        self.vocab = self.tokenizer.get_vocab()
        self.id2word = {v: k for k, v in self.vocab.items()}

    def gen_stream(self, config):
        contexts = config["prompt"]
        top_k = config.get("top_k_per_token", 20)
        top_p = config.get("top_p", 0.9)
        temperature = config.get("temperature", 0.9)
        seed = config.get("seed", 0)
        max_new_tokens = config.get("max_new_tokens", 512)
        gene_time = config.get("time", 15)
        history = config.get("history", [])
        do_sample = config.get("do_sample", True)
        convo_template = config.get("template", "aquila-v2")

        if seed > 0:
            torch.random.manual_seed(seed)

        MAX_LENGTH = 4096
        inputs = self.tokenizer.encode_plus(contexts)
        tokens = covert_prompt_to_input_ids_with_history(
            contexts,
            history=history,
            tokenizer=self.tokenizer,
            max_token=MAX_LENGTH,
            convo_template=convo_template,
        )
        del inputs["attention_mask"]
        inputs["input_ids"] = torch.tensor(tokens)[None,].to(self.device)
        streamer = TextIteratorStreamer(tokenizer=self.tokenizer,skip_prompt=True)
        # Arguments
        generation_kwargs = copy.deepcopy(self.generation_cfg)

        generation_kwargs.update(
            dict(
                inputs,
                streamer=streamer,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_time=gene_time,
                do_sample=do_sample,
            )
        )

        print(f"generation_kwargs: {generation_kwargs}")
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        return streamer


# model_info = {
#     "name": "aquilachat-7b",
#     "pretrained_model_name": "aquilachat2-7b",
#     "local_model_path": "/share/project/shixiaofeng/data/model_hub/Aquila2/ckpt_input/aquilachat2-7b",
#     "provides": "Aquila",
# }

# pipe = AquliaModel(model_info)

# config = {"prompt": "北京的景点有哪些？"}

# streamer = pipe.gen_stream(config)
# generated_text = ""
# for new_text in streamer:
#     generated_text += new_text
#     print(generated_text)
