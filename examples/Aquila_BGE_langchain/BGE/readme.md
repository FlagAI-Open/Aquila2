
## 安装环境

请按照下面的指示安装必要的库和依赖。

```
pip install -r requirements.txt
```

## 数据预处理

首先将自己的数据文件放到路径中，这里我们提供了一些示例数据，可见于data_demo.json，执行以下命令：

```bash
./preprocess.sh
```
会进行一系列预处理，包括：对data_demo.json进行过滤，生成abstract和meta的embedding，生成abstract和meta的BM25 index。

##启动

然后可以运行app.py，如果之前修改了保存路径，可以给app.py传新的参数

```python
python app.py \
--data-path data_demo_filter.json \
--abstract-emb-path abstract.npy \
--abstract-index-path abstract.index \
--abstract-bm25-index-path abstract_bm25_index \
--meta-emb-path meta.npy \
--meta-index-path meta.index \
--meta-bm25-index-path meta_bm25_index \
--batch-size 128
```

