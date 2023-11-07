## 安装环境

* python 版本 >= 3.10
* pytorch 版本 >= 2.0.1
* transformers 版本 >= 4.32.1
* CUDA 版本 >= 11.4 (GPU用户、flash-attention用户等需考虑此选项)

## 快速使用

### 环境配置

使用镜像文件：对于满足这个要求的环境，您也可以通过直接下载[docker]()文件并安装来配置所需的环境。

### 数据准备

我们提供了一些示例数据，可见于data_demo.json。您可根据自己的实际情况更换自己的数据。

### 模型下载

若实现在本地或离线环境下运行项目，需要首先将项目所需的模型下载至本地。本项目中默认使用的LLM模型[Aquilachat2-34B](https://model.baai.ac.cn/models)与Embedding模型[BGE](https://huggingface.co/BAAI/bge-large-en-v1.5)(https://huggingface.co/BAAI/bge-reranker-large) 。

### 构建本地知识库

数据预处理，包括：对data_demo.json进行过滤，生成abstract和meta的embedding，生成abstract和meta的BM25 index。执行一下命令：

```bash
./preproces.sh
```

### 微调

具体过程请参考https://github.com/FlagAI-Open/Aquila2/

### 一键启动

```
CUDA_VISIBLE_DEVICES=0 python local_search.py
```


