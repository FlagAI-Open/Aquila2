python preprocess_dataset.py --input-path ../data_demo.json --output-path ../data/ai_filter.json

python inference_abstract_emb.py --input-path ../data/ai_filter.json --output-path ../data/abstract.npy --batch-size 512

python inference_meta_emb.py --input-path ../data/ai_filter.json --output-path ../data/meta.npy --batch-size 512

mkdir meta_collection
mkdir abstract_collection

python build_bm25_data.py --data-path ../data/ai_filter.json --abstract-document-path abstract_collection --meta-document-path meta_collection

mkdir meta_bm25_index
mkdir abstract_bm25_index

python -m pyserini.index.lucene   --collection JsonCollection   --input meta_collection  --index meta_bm25_index   --generator DefaultLuceneDocumentGenerator   --threads 8   --storePositions --storeDocvectors --storeRaw
python -m pyserini.index.lucene   --collection JsonCollection   --input abstract_collection  --index abstract_bm25_index   --generator DefaultLuceneDocumentGenerator   --threads 8   --storePositions --storeDocvectors --storeRaw
