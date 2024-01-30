# !/usr/bin/env python
# -*- coding:utf-8 -*-
# ==================================================================
# [Author]       : shixiaofeng
# [Descriptions] :
# ==================================================================
import argparse
import sys
from flask import Flask, request, jsonify
from fastapi import FastAPI, Request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from models.bge.tool import SearchTool
import json

app = Flask(__name__)
search_tool = None
limiter = Limiter(get_remote_address, app=app, default_limits=["50 per minute"])


@app.route("/debug", methods=["GET"])
def debug():
    return "RETRIVAL SERVER IS SUCCESS"


@app.route("/search", methods=["POST"])
def index():
    global search_tool
    data = request.get_json()
    if isinstance(data, str):
        data = json.loads(data)

    input_text = data.get("query", None)
    retrieval_type = data.get("retrieval_type", "semantic")
    query_type = data.get("query_type", "by query")
    target_type = data.get("target_type", "original")
    num = min(int(data.get("num", 5)), 100)
    rerank = data.get("rerank", "disable")
    rerank_num = min(int(data.get("rerank_num", 50)), 200)
    if input_text is None:
        return jsonify({"error": "No query provided"}), 400
    try:
        print(
            jsonify(
                search_tool.search(
                    input_text,
                    retrieval_type,
                    query_type,
                    target_type,
                    num,
                    rerank,
                    rerank_num,
                )
            )
        )
        return (
            jsonify(
                search_tool.search(
                    input_text,
                    retrieval_type,
                    query_type,
                    target_type,
                    num,
                    rerank,
                    rerank_num,
                )
            ),
            200,
        )
    except Exception as e:
        return jsonify({"error": e}), 400


if len(sys.argv) == 2:
    BGE_DATA_PATH_BASE = sys.argv[1]
else:
    BGE_DATA_PATH_BASE = "models/bge/knowledge_data/db_content"
BGE_DATA_PATH = f"{BGE_DATA_PATH_BASE}/data_filter.json"
BGE_ABSTRACT_EMB_PATH = f"{BGE_DATA_PATH_BASE}/abstract.npy"
BGE_ABSTRACT_INDEX_PATH = f"{BGE_DATA_PATH_BASE}/abstract.index"
BGE_ABSTRACT_BM25_INDEX_PATH = f"{BGE_DATA_PATH_BASE}/abstract_bm25_index"
BGE_META_EMB_PATH = f"{BGE_DATA_PATH_BASE}/meta.npy"
BGE_META_INDEX_PATH = f"{BGE_DATA_PATH_BASE}/meta.index"
BGE_META_BM25_INDEX_PATH = f"{BGE_DATA_PATH_BASE}/meta_bm25_index"
BGE_BATCH_SIZE = 128

search_tool = SearchTool(
    BGE_DATA_PATH,
    BGE_ABSTRACT_EMB_PATH,
    BGE_ABSTRACT_INDEX_PATH,
    BGE_ABSTRACT_BM25_INDEX_PATH,
    BGE_META_EMB_PATH,
    BGE_META_INDEX_PATH,
    BGE_META_BM25_INDEX_PATH,
    BGE_BATCH_SIZE,
)

app.run("0.0.0.0", 5678)
