首先，将ai.json放到路径内，然后运行preprocess.sh

会进行一系列预处理，包括：

1. 对ai.json进行过滤

2. 生成abstract和meta的embedding

3. 生成abstract和meta的BM25 index

如果想修改其中的保存路径，可以编辑preprocess.sh修改

```shell
bash preprocess.sh
```

然后可以运行app.py，如果之前修改了保存路径，可以给app.py传新的参数

```shell
python app.py \
--data-path ai_filter.json \
--abstract-emb-path abstract.npy \
--abstract-index-path abstract.index \
--abstract-bm25-index-path abstract_bm25_index \
--meta-emb-path meta.npy \
--meta-index-path meta.index \
--meta-bm25-index-path meta_bm25_index \
--batch-size 128

```

