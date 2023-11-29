# Usage: deepspeed train_lora.py --deepspeed <$PATH_TO_DEEPSPEED_CONFIG>

# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
from dataclasses import dataclass, field
import logging
import pathlib
import typing
import os
import json
import math
from typing import Dict, Optional, Sequence
import numpy as np
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import transformers
from transformers import Trainer, BitsAndBytesConfig, deepspeed
import torch

from flagai.model.aquila2.aquila2_flash_attn_monkey_patch import (
    replace_aquila_attn_with_flash_attn, )

from torch.utils.data import Dataset
from transformers.trainer_pt_utils import LabelSmoother

from flagai.model.aquila2.conversation import SeparatorStyle
from flagai.model.aquila2.conversation import get_conversation_template

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    use_sequential_init: bool = False
    cache_dir: typing.Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    flash_attn: bool = False
    use_lora: bool = False
    use_single_node: bool = False

@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    lazy_preprocess: bool = False
    convo_template: str = field(
        default='aquila', metadata={"help": "Template of datasets."}
    )

@dataclass
class ModelArguments:
    model_dir: Optional[str] = field(default="./checkpoints")
    model_name: Optional[str] = field(default="aquila2chat-hf")


@dataclass
class LoraArguments:
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: typing.List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"])
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""
    if deepspeed.is_deepspeed_zero3_enabled():
        state_dict = trainer.model_wrapped._zero3_consolidated_16bit_state_dict(
        )
    else:
        state_dict = trainer.model.state_dict()

    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def preprocess(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    convo_template,
) -> Dict:
    conv = get_conversation_template(convo_template)
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()

    assert conv.sep_style == SeparatorStyle.ADD_COLON_TWO or \
           conv.sep_style == SeparatorStyle.NO_COLON_TWO

    # Mask targets. Only compute loss on the assistant outputs.
    if conv.sep_style == SeparatorStyle.NO_COLON_TWO:
        sep = conv.sep + conv.roles[1]
    else:
        sep = conv.sep + conv.roles[1] + ":"

    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        turns = conversation.split(conv.sep2)
        cur_len = 0
        for i, turn in enumerate(turns):
            if turn == "":
                break
            turn_len = len(tokenizer(turn).input_ids)

            parts = turn.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            instruction_len = len(tokenizer(parts[0]).input_ids)

            # Ignore the user instructions
            target[cur_len:cur_len + instruction_len] = IGNORE_TOKEN_ID
            cur_len += instruction_len
            cur_len += len(tokenizer(parts[1]).input_ids)
            cur_len += 1

        target[cur_len:] = IGNORE_TOKEN_ID

        if False:  # Inspect and check the correctness of masking
            z = target.clone()
            z = torch.where(z == IGNORE_TOKEN_ID, tokenizer.unk_token_id, z)
            rank0_print(tokenizer.decode(z))

        if cur_len < tokenizer.model_max_length:
            # Ignore cases
            if cur_len != total_len:
                target[:] = IGNORE_TOKEN_ID
                rank0_print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)")

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer,
                 convo_template):
        super(SupervisedDataset, self).__init__()

        rank0_print("Formatting inputs...")
        sources = [example["conversations"] for example in raw_data]
        data_dict = preprocess(sources, tokenizer, convo_template)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]
        self.convo_template = convo_template

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
        )


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer,
                 convo_template):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.cached_data_dict = {}
        self.convo_template = convo_template

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        ret = preprocess([self.raw_data[i]["conversations"]], self.tokenizer,
                         self.convo_template)
        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            attention_mask=ret["attention_mask"][0],
        )
        self.cached_data_dict[i] = ret

        return ret


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = (LazySupervisedDataset
                   if data_args.lazy_preprocess else SupervisedDataset)
    rank0_print("Loading data...")

    if data_args.data_path.endswith('jsonl'):
        import jsonlines
        train_json = []
        with jsonlines.open(data_args.data_path) as reader:
            for json_obj in reader:
                _key = 'conversations'
                if _key in json_obj and len(json_obj[_key]) > 0:
                    train_json.append(json_obj)
    else:
        train_json = json.load(open(data_args.data_path, "r"))
    train_dataset = dataset_cls(train_json,
                                tokenizer=tokenizer,
                                convo_template=data_args.convo_template)

    if data_args.eval_data_path:
        if data_args.eval_data_path.endswith('jsonl'):
            import jsonlines
            eval_json = []
            with jsonlines.open(data_args.eval_data_path) as reader:
                _key = 'conversations'
                if _key in json_obj and len(json_obj[_key]) > 0:
                    eval_json.append(json_obj)
        else:
            eval_json = json.load(open(data_args.eval_data_path, "r"))
        eval_dataset = dataset_cls(eval_json,
                                   tokenizer=tokenizer,
                                   convo_template=data_args.convo_template)
    else:
        eval_dataset = None

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)


def maybe_zero_3(param):
    if hasattr(param, "ds_id"):
        assert param.ds_status == ZeroParamStatus.NOT_AVAILABLE
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {
            k: t
            for k, t in named_params if "lora_" in k or "bias" in k
        }
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v) for k, v in to_return.items()}
    return to_return


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments))
    (
        model_args,
        data_args,
        training_args,
        lora_args,
    ) = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    if training_args.flash_attn:
        replace_aquila_attn_with_flash_attn()

    device_map = None
    if training_args.use_lora:
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        ddp = world_size != 1
        # if lora_args.q_lora:
        device_map = {
            "": int(os.environ.get("LOCAL_RANK") or 0)
        } if ddp else None
        if len(training_args.fsdp) > 0 or deepspeed.is_deepspeed_zero3_enabled(
        ):
            logging.warning(
                "FSDP and ZeRO3 are both currently incompatible with QLoRA.")

    if training_args.use_single_node:
        device_map = 'auto'

    compute_dtype = (torch.float16 if training_args.fp16 else
                     (torch.bfloat16 if training_args.bf16 else torch.float32))

    from flagai.auto_model.auto_loader import AutoLoader
    autoloader = AutoLoader("aquila2",
                        model_dir=model_args.model_dir,
                        model_name=model_args.model_name,
                        inference_mode=False,
                        model_max_length=training_args.model_max_length,
                        cache_dir=training_args.cache_dir,
                        device_map=device_map,
                        quantization_config=BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_use_double_quant=True,
                            bnb_4bit_quant_type="nf4",
                            bnb_4bit_compute_dtype=compute_dtype,
                        ) if lora_args.q_lora else None,
                    )
    model = autoloader.get_model()

    if training_args.use_lora:
        lora_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            target_modules=lora_args.lora_target_modules,
            lora_dropout=lora_args.lora_dropout,
            bias=lora_args.lora_bias,
            task_type="CAUSAL_LM",
        )

        if lora_args.q_lora:
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=training_args.gradient_checkpointing
            )
            if not ddp and torch.cuda.device_count() > 1:
                # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
                model.is_parallelizable = True
                model.model_parallel = True

        model = get_peft_model(model, lora_config)
        if training_args.deepspeed is not None and training_args.local_rank == 0:
            model.print_trainable_parameters()
    if training_args.flash_attn:
        for name, module in model.named_modules():
            if "norm" in name:
                module = module.to(compute_dtype)
            if "lm_head" in name or "embed_tokens" in name:
                if hasattr(module, "weight"):
                    module = module.to(compute_dtype)

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        os.path.join(model_args.model_dir,model_args.model_name),
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
    )
    tokenizer.pad_token = tokenizer.unk_token

    # Load data
    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args)

    # Start trainner
    trainer = Trainer(model=model,
                      tokenizer=tokenizer,
                      args=training_args,
                      **data_module)

    # model.config.use_cache = False

    if training_args.use_sequential_init:
        # Inform next rank
        if local_rank < local_world_size - 2:
            print(
                f"Send message from rank {global_rank} to rank {global_rank+1}"
            )
            torch.distributed.send(tensor, global_rank + 2)
    model.config.use_cache = False
    if list(pathlib.Path(training_args.output_dir).glob(
            "checkpoint-*")) and not lora_args.q_lora:
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    # check if zero3 mode enabled
    if deepspeed.is_deepspeed_zero3_enabled():
        # use deepspeed engine internal function to gather state dict
        # state_dict_zero3 contains whole parameters of base and lora adapters
        # we will not extract lora parameters since peft save_pretrained will do that
        # https://github.com/huggingface/peft/blob/3714aa2fff158fdfa637b2b65952580801d890b2/src/peft/peft_model.py#L125
        # https://github.com/huggingface/peft/blob/3714aa2fff158fdfa637b2b65952580801d890b2/src/peft/utils/save_and_load.py#L19
        state_dict_zero3 = trainer.model_wrapped._zero3_consolidated_16bit_state_dict(
        )
        if training_args.local_rank == 0:
            state_dict = state_dict_zero3
    else:
        # in other mode we use original code from fastchat team, to make sure our change is minimum
        state_dict = get_peft_state_maybe_zero_3(model.named_parameters(),
                                                 lora_args.lora_bias)

    model.config.use_cache = True
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer,
                                   output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
