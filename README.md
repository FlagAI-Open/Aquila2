<p align="left">
    <a href="README_CN.md">中文</a>&nbsp ｜ &nbspEnglish&nbsp
</p>
<br><br>

<p align="center">
    <img src="assets/logo.png" width="400"/>
<p>
<br>

<p align="center">
    <a href="https://model.baai.ac.cn/models">BAAI ModelHub</a>&nbsp&nbsp<a href="https://huggingface.co/BAAI">Hugging Face</a>&nbsp&nbsp<a href="assets/wechat-qrcode.jpg">WeChat (微信)</a>&nbsp&nbsp
<br>
</p>
<br>

We opensource our **Aquila2** series, now including **Aquila2**, the base language models, namely **Aquila2-7B** and **Aquila2-34B**, as well as **AquilaChat2**, the chat models, namely **AquilaChat2-7B** and **AquilaChat2-34B**.

| Model Name         | Modelhub  | Huggingface | 
|----------------------|:----:|:-----------: |
| Aquila2-7B | https://model.baai.ac.cn/model-detail/100118 |    -     | 
| AquilaChat2-7B | https://model.baai.ac.cn/model-detail/100117 |   -      | 
| Aquila2-34B | https://model.baai.ac.cn/model-detail/100119  |    -    | 
| AquilaChat2-34B | https://model.baai.ac.cn/model-detail/100116 |   -      |

In this repo, you can figure out:

* Quickstart with Aquila2, and enjoy the simple inference.
* Tutorials on finetuning, including full-parameter tuning, LoRA, and Q-LoRA.
* Statistics of long-context understanding evaluation
* License agreement
* ...

Feel free to shoot us issues (better in English so that more people can understand you)! If you would like to help us, send us pull requests with no hesitation! We are always excited about PR! 
<br>

## News and Updates

* 2023.10.10 🔥 We release **Aquila2-34B** and **AquilaChat2-34B** on BAAI ModelHub and Hugging Face.
<br>

## Performance

Aquila2-34B and Aquila2-7B outperform the baseline models of similar model sizes on a series of benchmark datasets, e.g., MMLU, C-Eval, GSM8K, MATH, HumanEval etc., which evaluate the models' capabilities on natural language understanding, mathematic problem solving, coding, etc. 

|      Model      | C-Eval |  MMLU  | CMMLU  | GSM8K  | GaoKao |  MATH  | HumanEval | WMT22 (en-zh) | WinoGrande |
| :-------------: | :----: | :----: | :----: | :----: | :----: | :----: | :-------: | :-----------: | :--------: |
|                 | 5-shot | 5-shot | 5-shot | 8-shot |        | 4-shot |  0-shot   |    0-shot     |   0-shot   |
|   InternLM-7B   |  48.6  |  51.2  |  51.8  |  31.2  |  49.6  |  6.3   |   13.4    |     53.3      |    68.2    |
|  InternLM-20B   |  53.7  |  61.8  |  59.0  |  52.6  |  63.6  |  7.9   |   25.6    |     56.9      |    75.1    |
|   ChatGLM2-6B   |  51.7  |  47.9  |  48.8  |  32.4  |  49.4  |  6.5   |    9.2    |     45.7      |            |
|  ChatGLM2-12B   |  61.6  |  56.2  |        |  40.9  |        |        |           |               |            |
|  Baichuan2-7B   |  52.3  |  54.6  |  57.1  |  24.5  |  53.6  |  5.6   |   18.3    |     55.9      |    68.4    |
|  Baichuan2-13B  |  55.6  |  56.9  |  62.0  |  52.8  |  59.7  |  10.1  |   17.1    |     60.5      |    70.3    |
|     Qwen-7b     |  56.7  |  58.0  |  62.2  |  51.7  |  58.5  |  6.5   |   29.9    |     58.1      |    66.1    |
|    Qwen-14b     |  71.4  |  65.8  |  70.5  |  58.7  |  65.4  |  13.4  |   32.3    |     55.0      |    67.4    |
|    LLaMA2-7B    |  34.1  |  46.9  |  31.4  |  16.2  |  41.7  |  3.2   |   12.8    |     36.4      |    67.1    |
|   LLaMA2-70B    |  52.1  |  69.5  |        |  56.8  |  64.5  |  13.5  |   29.9    |               |    78.0    |
| **Aquila2-7B**  |  48.9  |  54.9  |  56.1  |  41.9  |  54.0  |  10.9  |   21.4    |     57.3      |    67.5    |
| **Aquila2-33B** |  62.2  |  60.0  |  65.9  |  56.3  |  64.6  |  11.6  |   25.3    |     60.0      |    70.6    |

<br>

### Long Context Performance
|           Model           |       Method       | SingleQA | MultiQA | Summarization | Code Completion | Few Shot | Synthetics | Selection | Other |
| :-----------------------: | :----------------: | :------: | :-----: | :-----------: | :-------------: | :------: | :--------: | :-------: | :---: |
|     GPT-3.5-Turbo-16K     |      Unknown       |   47.6   |  36.2   |     23.0      |      54.5       |   77.5   |    27.5    |   33.6    | 35.3  |
|   LongChat-7B-v1.5-32K    |       PI+SFT       |   27.5   |  20.3   |     22.5      |      57.0       |   62.9   |    18.8    |   21.7    | 23.7  |
|      ChatGLM2-6B-32K      | Continual Pretrain |   44.1   |  34.7   |     20.8      |      52.7       |   68.7   |    23.6    |   30.8    | 31.6  |
|  Baichuan2-13B-Chat-16K   |        None        |   14.5   |  12.6   |     14.0      |                 |   22.2   |    11.9    |   11.6    | 14.7  |
| Qwen-14B-Chat-dynamic-ntk |    Dynamic NTK     |   24.4   |  20.2   |     22.3      |      37.3       |   47.2   |    18.6    |   16.1    | 24.0  |
|     Internlm-20B-Chat     |        None        |   19.2   |  17.5   |     16.7      |      26.0       |   41.0   |    16.5    |   16.6    | 19.7  |
|    **Aquila2-7B-16K**     |       PI+SFT       |   21.8   |  22.4   |     19.1      |      22.8       |   50.1   |    18.4    |   29.5    | 25.0  |
|    **Aquila2-33B-16K**    |       PI+SFT       |          |         |               |                 |          |            |   31.7    | 29.5  |

<br>

### Reasoning Performance
|      Model      | Placeholder |
| :-------------: | :---------: |
|     ChatGPT     |             |
|      GPT-4      |             |
|  InternLM-20B   |             |
|  Baichuan2-7B   |             |
|  Baichuan2-13B  |             |
|     Qwen-7b     |             |
|    Qwen-14b     |             |
| **Aquila2-7B**  |             |
| **Aquila2-33B** |             |

<br>

## Requirements

* python 3.10 and above
* pytorch 1.12 and above, 2.0 and above are recommended
* transformers 4.32 and above
* CUDA 11.4 and above are recommended (this is for GPU users, flash-attention users, etc.)
<br>

## Quickstart
We have provided a straightforward example to illustrate how to quickly get started with Aquila2.

Before proceeding, ensure that your environment is properly configured and that the necessary packages have been installed. First and foremost, ensure that these prerequisites are met and then follow the instructions below to install the necessary libraries and dependencies.
```
pip install -r requirements.txt
```

If your device supports fp16 or bf16 precision, we also recommend installing flash-attention to enhance execution speed and reduce memory consumption. It's important to note that flash-attention is optional, and the project can be executed normally without it.

For the installation of flash-attention, please refer to https://github.com/Dao-AILab/flash-attention/.

### ModelHub

You can now utilize the AquilaChat2-7B model for inference as follows:

```python
from flagai.auto_model.auto_loader import AutoLoader

# Model name
model_name = 'AquilaChat2-7B'
# model_name = 'AquilaChat2-34B'

# Load the model and tokenizer
autoloader = AutoLoader("aquila2", model_name=model_name)
# To modify the model loading path, use the model_dir parameter
# autoloader = AutoLoader("aquila2", model_dir='./checkpoints', model_name=model_name)
# To load the LoRA module, you need to provide the path to the LoRA module
# autoloader = AutoLoader("aquila2", model_name=model_name，lora_dir='./examples/checkpoints/lora/aquila2chat')
# To load the LoRA module, you need to provide the path to the LoRA module
# autoloader = AutoLoader("aquila2", model_name=model_name，qlora_dir='./examples/checkpoints/qlora/aquila2chat')

model = autoloader.get_model()
tokenizer = autoloader.get_tokenizer()

# 对话测试样例
test_data = [
    "Write a tongue twister that's extremely difficult to pronounce.",
]

for text in test_data:
    print(model.predict(text, tokenizer=tokenizer))
```

The results of our execution are as follows:

```
Harry had a harpy flight, Fred had a fiddle, and George had a gecko for breakfast.  Say that three times fast and see how long you can make it last!
```
### Base Model Inference

The distinction between the basic model inference and the dialogue model is that it requires setting sft=False during the model inference.

```
from flagai.auto_model.auto_loader import AutoLoader

# Model name
model_name = 'Aquila2-7B'
# model_name = 'Aquila2-34B'

# Load the model and tokenizer
autoloader = AutoLoader("aquila2", model_name=model_name)

model = autoloader.get_model()
tokenizer = autoloader.get_tokenizer()

# Example
test_data = [
    "Write a tongue twister that's extremely difficult to pronounce.",
]

for text in test_data:
    print(model.predict(text, tokenizer=tokenizer, sft=False))
```

<br>

### Hugging Face

To use AquilaChat2-7B for the inference, all you need to do is to input a few lines of codes as demonstrated below. 

```python
```
<br>

## Quantization

### Usage

```python
import torch 
from flagai.auto_model.auto_loader import AutoLoader
from transformers import BitsAndBytesConfig

model_name = 'AquilaChat2-7B'

autoloader = AutoLoader("aquila2", model_name=model_name, 
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    ))

model = autoloader.get_model()
tokenizer = autoloader.get_tokenizer()
# 

test_data = [
    "Write a tongue twister that's extremely difficult to pronounce.",
]

for text in test_data:
    print(model.predict(text, tokenizer=tokenizer))

```

### Inference Speed

We measured the average inference speed (tokens/s) of generating 2048 and 8192 tokens under BF16 precision and Int4 quantization, respectively.

| Quantization         | Speed (2048 tokens) | Speed (8192 tokens) |
|----------------------|:-------------------:|:-------------------:|
| Aquila2-34B-Chat (BF16) |        30.70        |        21.73        |
| Aquila2-34B-Chat (Int4) |        37.11        |        26.11        |

In detail, the setting of profiling is generating 8192 new tokens with 1 context token. The profiling runs on a single A100-SXM4-80G GPU with PyTorch 2.0.1 and CUDA 11.4. The inference speed is averaged over the generated 8192 tokens.

### GPU Memory Usage

We also profile the peak GPU memory usage for encoding 2048 tokens as context (and generating single token) and generating 8192 tokens (with single token as context) under BF16 or Int4 quantization level, respectively. The results are shown below.

| Quantization         | Peak Usage for Encoding 2048 Tokens | Peak Usage for Generating 8192 Tokens |
|----------------------|:-----------------------------------:|:-------------------------------------:|
| Aquila2-34B-Chat (BF16) |               30.15GB               |                38.94GB                |
| Aquila2-34B-Chat (Int4) |               13.00GB               |                21.79GB                |

<br>

## Pretraining
### Usage
<br>

## Finetuning

### Usage
We provide users with a series of fine-tuning scripts designed to adapt models to various downstream tasks using custom data. Within the comments section of the scripts, users will find detailed instructions indicating which parameters may need adjustments based on specific needs.

Before initiating the fine-tuning process, you are required to have your training data prepared. All samples should be consolidated into a list and stored in a json file. Each sample should be represented as a dictionary, encompassing an ID and conversation, with the latter presented in list format. Below is an example for your reference:

```json
{"id": "alpaca_data.json_1", "conversations": [{"from": "human", "value": "What are the three primary colors?"}, {"from": "gpt", "value": "The three primary colors are red, blue, and yellow."}], "instruction": ""}
```
Subsequently, you can utilize the variety of fine-tuning scripts we offer for different purposes:

- Execute `finetune/7B/finetune.sh` for a full parameter fine-tuning of the 7B model
- Execute `finetune/7B/finetune_lora.sh` for LoRA fine-tuning of the 7B model
- Execute `finetune/7B/finetune_qlora.sh` for Q-LoRA fine-tuning of the 7B model
- Execute `finetune/34B/finetune.sh` for a full parameter fine-tuning of the 34B model
- Execute `finetune/34B/finetune_lora.sh` for LoRA fine-tuning of the 34B model
- Execute `finetune/34B/finetune_qlora.sh` for Q-LoRA fine-tuning of the 34B model

### Optimization Effects

Below are the data on memory usage and training speed for the 7B and 34B models using full-parameter fine-tuning, LoRA, and QLoRA with different input lengths. The evaluation was conducted on a machine equipped with an A100-SXM4-80G GPU, utilizing CUDA 12.1 and Pytorch 2.1. The input length for the 7B model is 2048, and for the 34B model, it is 4096. All tests were performed using a batch size of 4 and a gradient accumulation of 1, and both memory usage (in GB) and training speed (in s/iter) were recorded. The specific data is as follows:

<table>
    <tr>
      <th>Model Size</th><th>Method</th><th>Memory</th><th>speed</th>
    </tr>
    <tr>
        <th rowspan="3">7B</th><td>SFT</td><td>43.9G</td><td>2.67s/iter</td>
    </tr>
    <tr>
        <td>LoRA</td><td>29.4G</td><td>2.04s/iter</td>
    </tr>
    <tr>
        <td>Q-LoRA</td><td>19.9G</td><td>2.14s/iter</td>
    </tr>
    <tr>
        <th rowspan="1">34B</th><td>Q-LoRA</td><td>37.7G</td><td>8.22s/iter</td>
    </tr>
</table>

<br>

## Demo

### Web UI
Our web will be open soon.

<br>

## Long-Context Understanding

To extend the context length and break the bottleneck of training sequence length, we introduce several techniques, including NTK-aware interpolation, window attention, and LogN attention scaling.

## Tokenizer

Our tokenizer of BBPE type is trained on a 50GB corpus, mainly sampled from deduplicated Pile and deduplicated WuDao contents. We also add some special tokens for passage and conversation separation.
<br>

## Reproduction

For your reproduction of the model performance on benchmark datasets, we provide scripts for you to reproduce the results. Check [eval/README.md](eval/README.md) for more information. Note that the reproduction may lead to slight differences from our reported results.
<br>

## License Agreement

<br>

## Contact Us

If you are interested to leave a message to either our research team or product team, join our WeChat groups!

