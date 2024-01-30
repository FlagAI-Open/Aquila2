#!/bin/bash
# ---------------------------------------------------------------
# [File]         : preprocess.sh
# [CreatedDate]  : Thursday, 1970-01-01 08:00:00
# [Author]       : shixiaofeng
# [Descriptions] :
# ==================================================================
# [ChangeLog]:
# [Date]    	[Author]	[Comments]
# ------------------------------------------------------------------
set -u

CUDA_DEVICE=${1:-0}
BATCH_SIZE=${2:-256}

SRC_DIR=./knowledge_data
SRC_FILENAME=data_demo.jsonl
DB_DIR=$SRC_DIR/db_content

echo "process: $SRC_DIR/$SRC_FILENAME"

rm -r $DB_DIR
mkdir -p $DB_DIR

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 preprocess_dataset.py --input-path $SRC_DIR/$SRC_FILENAME --output-path $DB_DIR/data_filter.json

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 inference_abstract_emb.py --input-path $DB_DIR/data_filter.json --output-path $DB_DIR/abstract.npy --batch-size $BATCH_SIZE
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 inference_meta_emb.py --input-path $DB_DIR/data_filter.json --output-path $DB_DIR/meta.npy --batch-size $BATCH_SIZE

mkdir -p $DB_DIR/meta_collection
mkdir -p $DB_DIR/abstract_collection

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 build_bm25_data.py --data-path $DB_DIR/data_filter.json --abstract-document-path $DB_DIR/abstract_collection --meta-document-path $DB_DIR/meta_collection

mkdir $DB_DIR/meta_bm25_index
mkdir $DB_DIR/abstract_bm25_index

python3 -m pyserini.index.lucene --collection JsonCollection --input $DB_DIR/meta_collection --index $DB_DIR/meta_bm25_index --generator DefaultLuceneDocumentGenerator --threads 2 --storePositions --storeDocvectors --storeRaw
python3 -m pyserini.index.lucene --collection JsonCollection --input $DB_DIR/abstract_collection --index $DB_DIR/abstract_bm25_index --generator DefaultLuceneDocumentGenerator --threads 2 --storePositions --storeDocvectors --storeRaw
