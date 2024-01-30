# !/usr/bin/env python
# -*- coding:utf-8 -*-
# ==================================================================
# [CreatedDate]  : Thursday, 1970-01-01 08:00:00
# [Author]       : shixiaofeng
# [Descriptions] :
# ==================================================================
# [ChangeLog]:
# [Date]    	[Author]	[Comments]
# ------------------------------------------------------------------

import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

global BGE_RETRIVAL, BGE_RERANK, MODEL_MAX_LENGTH

MODEL_MAX_LENGTH = 512

# BGE_RETRIVAL = "./checkpoints/bge-large-en-v1.5"
# QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

BGE_RETRIVAL = "/share/project/shixiaofeng/data/BGE/bge-large-zh-v1.5"
## 中文的指令
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："

BGE_RERANK = "/share/project/shixiaofeng/data/BGE/bge-reranker-large"

print("Loading retrial_model model ... ")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device is {device}")
retrial_tokenizer = AutoTokenizer.from_pretrained(BGE_RETRIVAL)
retrial_model = AutoModel.from_pretrained(BGE_RETRIVAL).to(device)
print("load retrival model done ")

print("Loading rerank_model model ... ")
rerank_tokenizer = AutoTokenizer.from_pretrained(BGE_RERANK)
rerank_model = AutoModelForSequenceClassification.from_pretrained(BGE_RERANK).to(device)
print("load rerank model done ")


