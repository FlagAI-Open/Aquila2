## Requirements

* python 3.10 and above
* pytorch 2.0.1 and above
* transformers 4.32.1 and above
* CUDA 11.4 and above are recommended (this is for GPU users, flash-attention users, etc.)

## Quickstart

### 1.Environment Configuration

For environments that meet this requirement, you can configure the required environment by directly downloading the [docker](https://model.baai.ac.cn/model-detail/220119) file and installing it.

### 2.Data Preparation

We have provided some sample data, which can be found in data_demo.json. You can change your data according to your actual situation.

### 3.Model Download

If you want to run the project in a local or offline environment, you need to first download the models required for the project to the local computer. The LLM model [Aquilachat2-34B](https://model.baai.ac.cn/models) and Embedding model [BGE](https://huggingface.co/BAAI/bge-large-en), [BGE-reranker](https://huggingface.co/BAAI/bge-reranker-large) are used by default in this project.

### 3.Build Local Knowledge Base

Data preprocessing includes: filtering data_demo.json, generating abstract and meta embeddings, and generating abstract and meta BM25 indexes. Execute the following command:

```bash
./preproces.sh
```

### 4.LLM Fine-tuning

For the specific process, please refer to https://github.com/FlagAI-Open/Aquila2/ .

### 5.Run

```
CUDA_VISIBLE_DEVICES=0 python local_search.py
```