import copy
import json
import logging
import random
import re
import sys
import time
import traceback
from typing import List


import numpy as np
import torch
from flagai.auto_model.auto_loader import AutoLoader
from flagai.data.tokenizer import Tokenizer


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
_LogFormat = logging.Formatter(
    "%(asctime)2s -%(name)-12s: %(levelname)-s/Line[%(lineno)d]/Thread[%(thread)d]  - %(message)s")

# create console handler with a higher log level
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(_LogFormat)
logger.addHandler(console)



class AquliaModel(object):
    def __init__(self, model_info):
        self.setup_model(model_info)

    def setup_model(self, model_info):
        model_dir = model_info["local_model_path"]
        model_name = model_info["name"]
        self.device = "cuda:0"
        start_time = time.time()
        logger.info(f"AquliaModel model_dir: [{model_dir}]")
        logger.info(f"AquliaModel model_name: [{model_name}]")

        loader = AutoLoader("aquila2",
                            model_dir=model_dir,
                            qlora_dir='../checkpoints/aquilachat2-34b-sft-aquila_experiment/checkpoint-747',
                            model_name=model_name,
                            use_cache=True,
                            fp16=True,
                            device=self.device,
                            all_devices=True)

        end_time = time.time()
        logger.info(f"Load model time: {end_time - start_time}s")

        self.model = loader.get_model()
        self.tokenizer = loader.get_tokenizer()
        self.vocab = self.tokenizer.get_vocab()
        self.id2word = {v: k for k, v in self.vocab.items()}
   
        

    
