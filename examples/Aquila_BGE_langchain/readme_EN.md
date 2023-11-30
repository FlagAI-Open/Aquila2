

## Introduction
This is a local knowledge base question-answer application based on the large language model Aquilachat2 and the vectorized retrieval model BGE supported by BAAI. The implementation principle of this project is shown in the figure below, including reading text -> text vectorization (BGE) and building a vector library -> query vectorization -> matching the top M documents most similar to the query - > Combining the matching documents and query to form a prompt -> Submitting it to Aquilachat2 to generate an answer.

![Image text](https://github.com/zll1995-nlp/Aquila2/blob/main/examples/Aquila_BGE_langchain/images/pic_2_EN.png)



## Requirements

* python 3.10 and above
* pytorch 2.0.1 and above
* transformers 4.32.1 and above
* CUDA 11.4 and above are recommended (this is for GPU users, flash-attention users, etc.)

## Quickstart

### 1.Environment Configuration

For environments that meet this requirement, you can configure the required environment by directly downloading the [docker](https://model.baai.ac.cn/model-detail/220119) file and installing it. Because of all already installed dependencies, in the container you just pull the source [FlagAI](https://github.com/FlagAI-Open/FlagAI.git) ,and add the path to environment variables export PYTHONPATH=$FLAGAI_HOME:$PYTHONPATH.

### 2.Data Preparation

We have provided some sample data, which can be found in data_demo.json. You can change your data according to your actual situation.

### 3.Model Download

If you want to run the project in a local or offline environment, you need to first download the models required for the project to the local computer. The LLM model [Aquilachat2-34B](https://model.baai.ac.cn/models) and Embedding model [BGE](https://huggingface.co/BAAI/bge-large-en), [BGE-reranker](https://huggingface.co/BAAI/bge-reranker-large) are used by default in this project.

### 4.Build Local Knowledge Base

Data preprocessing includes: filtering data_demo.json, generating abstract and meta embeddings, and generating abstract and meta BM25 indexes. Execute the following command:

```bash
cd BGE
./preproces.sh
```

### 5.LLM Fine-tuning

For the specific process, please refer to https://github.com/FlagAI-Open/Aquila2/ .

### 6.Run

```python
cd Aquila_local_search
CUDA_VISIBLE_DEVICES=0 python local_search.py
```
### 7.Startup Interface

The local port is set to 9172. If it starts normally, you will see the following interface:

![Image text](https://github.com/zll1995-nlp/Aquila2/blob/main/examples/Aquila_BGE_langchain/images/pic_3.png)

