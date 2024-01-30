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
import os
import logging
import uuid
import sys
import traceback

logging_file_path = "./rag_pipeline.log"
handlers = [
    # logging.FileHandler(logging_file_path),
    logging.StreamHandler(sys.stdout)
]
DEBUG = False
if DEBUG:
    level = logging.DEBUG
else:
    level = logging.INFO

logging.basicConfig(
    level=level, format="%(asctime)s [%(levelname)s] %(lineno)d %(message)s", handlers=handlers
)

logger = logging.getLogger(__name__)


# 向量搜索服务设置，资源消耗很小，本地起服务
SearchToolURL = "http://127.0.0.1:5678"

if not SearchToolURL:
    from models.bge.tool import SearchTool
    # 知识库存储地址
    BGE_DATA_PATH_BASE = "models/bge/knowledge_data/db_content"
    BGE_DATA_PATH = f"{BGE_DATA_PATH_BASE}/data_filter.json"
    BGE_ABSTRACT_EMB_PATH = f"{BGE_DATA_PATH_BASE}/abstract.npy"
    BGE_ABSTRACT_INDEX_PATH = f"{BGE_DATA_PATH_BASE}/abstract.index"
    BGE_ABSTRACT_BM25_INDEX_PATH = f"{BGE_DATA_PATH_BASE}/abstract_bm25_index"
    BGE_META_EMB_PATH = f"{BGE_DATA_PATH_BASE}/meta.npy"
    BGE_META_INDEX_PATH = f"{BGE_DATA_PATH_BASE}/meta.index"
    BGE_META_BM25_INDEX_PATH = f"{BGE_DATA_PATH_BASE}/meta_bm25_index"
    BGE_BATCH_SIZE = 128

    SearchToolClient = SearchTool(
        BGE_DATA_PATH,
        BGE_ABSTRACT_EMB_PATH,
        BGE_ABSTRACT_INDEX_PATH,
        BGE_ABSTRACT_BM25_INDEX_PATH,
        BGE_META_EMB_PATH,
        BGE_META_INDEX_PATH,
        BGE_META_BM25_INDEX_PATH,
        BGE_BATCH_SIZE,
    )
    print("bge搜索服务启动成功, 本地启动服务")
else:
    SearchToolClient = None
    print(f"bge搜索服务启动成功, 使用url形式：{SearchToolURL}")

BGE_SEARCH_NUM = 10
BGE_TOPK_NUM = 3

retrival_configs = {
    "retrieval_type": "merge",
    "query_type": "by query",
    "target_type": "conditional",
    "num": BGE_TOPK_NUM,
    "rerank": "enable",
    "rerank_num": BGE_SEARCH_NUM,
}


LLM_MODEL_NAME = "aquilachat2-34b"
LLM_MODEL_NAME = "aquilachat2-7b"
# 如果有链接，那么就不在本地起服务了，直接request调用
# LLM_URL = "http://120.92.91.62:9173"
LLM_URL = ""

llm_model_dict = {
    "aquilachat2-7b": {
        "name": "aquilachat-7b",
        "pretrained_model_name": "aquilachat2-7b",
        "local_model_path": "/share/project/shixiaofeng/data/model_hub/Aquila2/ckpt_input/aquilachat2-7b",
        "provides": "Aquila",
    },
    "aquilachat2-34b": {
        "name": "aquilachat2-34b",
        "pretrained_model_name": "aquilachat2-34b",
        "local_model_path": "./ckpt_input/AquilaChat2-34B-2023-10-24",
        "provides": "Aquila",
    },
}


LLM_CONFIG = {
    "prompt": "",
    "template": "aquila-v2",
    "seed": 1234,
    "top_k_per_token": 15,
    "top_p": 0.9,
    "temperature": 1.0,
    "sft": True,
    "max_new_tokens": 512,
    "gene_time": 25,
}



# LLM running device
LLM_DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
