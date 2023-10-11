<p align="left">
    <a href="README_CN.md">中文</a>&nbsp ｜ &nbspEnglish&nbsp
</p>
<br><br>

<p align="center">
    <img src="assets/logo.png" width="400"/>
<p>
<br>

<p align="center">
        🤗 <a href="https://huggingface.co/BAAI">Hugging Face</a>&nbsp&nbsp | &nbsp&nbsp <a href="https://model.baai.ac.cn/models">BAAI ModelHub</a>&nbsp&nbsp | &nbsp&nbsp <a href="assets/wechat-qrcode.png">WeChat(微信)</a>
</p>
<br>

We opensource our **Aquila2** series, now including **Aquila2**, the base language models, namely **Aquila2-7B** and **Aquila2-34B**, as well as **AquilaChat2**, the chat models, namely **AquilaChat2-7B** and **AquilaChat2-34B**.

| Model Name         | Download Sources  | 
|-------------------|:---------:|
| Aquila2-7B        | [<img src="assets/baai.png" width="14"/>](https://model.baai.ac.cn/model-detail/100118) 🤗|    -    | 
| AquilaChat2-7B    | [<img src="assets/baai.png" width="14"/>](https://model.baai.ac.cn/model-detail/100117) 🤗|    -    | 
| Aquila2-34B       | [<img src="assets/baai.png" width="14"/>](https://model.baai.ac.cn/model-detail/100119) 🤗|    -    | 
| AquilaChat2-34B   | [<img src="assets/baai.png" width="14"/>](https://model.baai.ac.cn/model-detail/100116) 🤗|    -    |

In this repo, you can figure out:

* Quickstart with Aquila2, and enjoy the simple inference.
* Tutorials on finetuning, including full-parameter tuning, LoRA, and Q-LoRA.
* Statistics of long-context understanding evaluation
* License agreement
* ...

Feel free to shoot us issues (better in English so that more people can understand you)! If you would like to help us, send us pull requests with no hesitation! We are always excited about PR! 
<br><br>

## News and Updates

* 2023.10.10 🔥 We release **Aquila2-34B** and **AquilaChat2-34B** on BAAI ModelHub and Hugging Face.
<br><br>

## Performance

Aquila2-34B and Aquila2-7B outperform the baseline models of similar model sizes on a series of benchmark datasets, e.g., MMLU, C-Eval, GSM8K, MATH, HumanEval etc., which evaluate the models' capabilities on natural language understanding, mathematic problem solving, coding, etc. 



TODO:

- Model name identity?
  - AquilaChat2-34B ? Aquila2-34B-Chat
  - 34B ? 33B
  - Chat ? chat
  - 7B ? 7b



### Base Model Performance

<br>

|      Model      | C-Eval |  MMLU  | CMMLU  | GSM8K  |  MATH  | HumanEval | WMT22 (en-zh) | WinoGrande |
| :-------------: | :----: | :----: | :----: | :----: | :----: | :-------: | :-----------: | :--------: |
|                 | 5-shot | 5-shot | 5-shot | 8-shot | 4-shot |  0-shot   |    0-shot     |   0-shot   |
|   InternLM-7B   |  48.6  |  51.2  |  51.8  |  31.2  |  6.3   |   13.4    |     53.3      |    68.2    |
|  InternLM-20B   |  53.7  |  61.8  |  59.0  |  52.6  |  7.9   |   25.6    |     56.9      |    75.1    |
|   ChatGLM2-6B   |  51.7  |  47.9  |  48.8  |  32.4  |  6.5   |    9.2    |     45.7      |            |
|  ChatGLM2-12B   |  61.6  |  56.2  |        |  40.9  |        |           |               |            |
|  Baichuan2-7B   |  52.3  |  54.6  |  57.1  |  24.5  |  5.6   |   18.3    |     55.9      |    68.4    |
|  Baichuan2-13B  |  55.6  |  56.9  |  62.0  |  52.8  |  10.1  |   17.1    |     60.5      |    70.3    |
|     Qwen-7b     |  56.7  |  58.0  |  62.2  |  51.7  |  6.5   |   29.9    |     58.1      |    66.1    |
|    Qwen-14b     |  71.4  |  65.8  |  70.5  |  58.7  |  13.4  |   32.3    |     55.0      |    67.4    |
|    LLaMA2-7B    |  34.1  |  46.9  |  31.4  |  16.2  |  3.2   |   12.8    |     36.4      |    67.1    |
|   LLaMA2-70B    |  52.1  |  69.5  |        |  56.8  |  13.5  |   29.9    |               |    78.0    |
| **Aquila2-7B**  |  48.9  |  54.9  |  56.1  |  41.9  |  10.9  |   21.4    |     57.3      |    67.5    |
| **Aquila2-33B** |  62.2  |  60.0  |  65.9  |  56.3  |  11.6  |   25.3    |     60.0      |    70.6    |

<br>

### Chat Model Performance

<br>

| Model              | Avg<br/>(obj. + subj.v2.0) | Avg<br/>(obj. + subj.v1.0) | Avg<br/>(obj.) | Avg<br/>(EN-obj.) | Avg<br/>(ZH-obj.) | Avg<br/>(ZH-subj.v2.0) | Avg<br/>(ZH-subj.v1.0) |
| :----------------- | :------------------------: | :------------------------: | :------------: | :---------------: | :---------------: | :--------------------: | :--------------------: |
| AquilaChat2-34B    |           70.2            |                            | 70.0 | 75.9 | 67.8 | 75.0 |                        |
| Baichuan2-13B-Chat |           64.3            |                            | 63.8 | 67.3 | 62.4 | 73.2 |                        |
| YuLan-Chat-2-13B   |           63.1            |                            | 63.1 | 69.8 | 60.2 | 63.3 |                        |
| InternLM-Chat-7B   |           61.1            | 61.5 | 61.7 | 62.4 | 61.4 | 50.2 | 58.1 |
| AquilaChat2-7B     |           60.2            |                            | 59.8 | 68.6 | 56.4 | 67.7 |                        |
| Baichuan2-7B-Chat  |           58.5            |                            | 57.9 | 62.1 | 56.4 | 67.9 |                        |
| InternLM-Chat-20B  |           53.8            |                            | 53.3 | 29.7 | 62.4 | 62.7 |                        |
| ChatGLM2-6B        |           35.3            | 35.7 | 34.2 | 43.7 | 30.2 | 54.2 | 62.1 |
| Qwen-14B-Chat      |           26.0            |                            | 23.2 | 23.1 | 23.0 | 77.4 |                        |
| Qwen-7B-Chat       |           13.0            | 13.4 | 0.0 | 0.0 | 0.0 | 67.4 | 75.4 |
| Baichuan-13B-Chat  |                           | 59.4 | 58.6 | 62.0 | 57.3 |                        | 73.3 |
| LLaMA-2-13B-Chat   |                           | 49.4 | 50.9 | 65.4 | 45.4 |                        | 22.0 |
| LLaMA-2-7B-Chat    |                           | 45.8 | 47.3 | 60.5 | 42.2 |                        | 18.3 |
| Alpaca             |                           | 43.2 | 43.2 | 58.4 | 36.9 |                        |                        |
| Ziya-LLaMA         |                           | 41.3 | 40.3 | 50.3 | 36.1 |                        | 59.5 |

<br>

### Long Context Performance

<br>

| Model                |   Method    | Avg. | ZH-Avg. | EN-Avg. | VCSUM(zh)<br>(Chinese) | LSHT(zh)<br>(Chinese) | HotpotQA<br>(English) | 2WikiMQA<br>(English) |
| :------------------- | :---------: | :--: | :-----: | :-----: | :--------------------: | :-------------------: | :-------------------: | :-------------------: |
| GPT-3.5-Turbo-16k    |      -      | 33.6 |  44.7   |  22.6   |          16.0          |         29.2          |         51.6          |         37.7          |
| AquilaChat2-34b-16k  |  PI + SFT   | 31.7 |  40.2   |  23.3   |          16.5          |         30.0          |         41.9          |         38.5          |
| ChatGLM2-6B-32k      |  PI + SFT   | 30.8 |  39.6   |  22.0   |          16.2          |         27.7          |         45.1          |         34.0          |
| AquilaChat2-7B-16k   |  PI + SFT   | 29.5 |  31.7   |  27.2   |          14.4          |         40.0          |         36.1          |         27.3          |
| InternLM-7B-8k       |      -      | 22.4 |  30.6   |  14.3   |          13.0          |         15.5          |         33.3          |         27.9          |
| ChatGLM2-6B          |    None     | 22.1 |  26.6   |  17.6   |          14.6          |         20.5          |         33.0          |         20.2          |
| LongChat-7B-v1.5-32k |  PI + SFT   | 21.7 |  26.1   |  17.4   |          14.0          |         20.8          |         31.5          |         20.6          |
| Baichuan2-7B-chat    |    None     | 21.3 |  25.9   |  16.8   |          13.6          |         20.0          |         32.8          |         18.9          |
| Internlm-20B-chat    |    None     | 16.6 |  24.3   |   8.9   |          11.9          |          6.0          |         24.4          |         24.2          |
| Qwen-14B-chat        | Dynamic NTK | 16.1 |  20.8   |  11.5   |          16.6          |          6.4          |         22.9          |         18.8          |
| XGen-7B-8k           |  Pre-train  | 16.0 |  21.3   |  10.8   |          1.5           |         20.0          |         14.2          |         28.3          |
| Llama2-7B-chat-4k    |    None     | 14.0 |  18.0   |  10.0   |          0.2           |         19.8          |         11.6          |         24.3          |
| Baichuan2-13B-chat   |    None     | 10.5 |  14.8   |   6.3   |          7.0           |          5.5          |         16.0          |         13.6          |

<br>

### Reasoning Tasks Performance

<br>

| Model                        | Avg. | bAbI#16<br>(Inductive) | CLUTRR<br>(Inductive) | bAbI#15<br>(Deductive) | EntailmentBank<br>(Deductive) | αNLI<br>(Abductive) | E-Care<br>(Casual) |
| :--------------------------- | :--: | :--------------------: | :-------------------: | :--------------------: | :---------------------------: | :-----------------: | :----------------: |
| Baichuan2-7B-Chat            | 47.8 |          40.0          |         26.7          |          43.3          |             73.3              |        53.3         |        50.0        |
| Qwen-7B-Chat                 | 49.5 |          20.0          |         10.0          |          66.7          |             86.7              |        56.7         |        56.7        |
| Qwen-14B-Chat                | 51.1 |          26.7          |         10.0          |          63.3          |             86.7              |        63.3         |        56.7        |
| Baichuan2-13B-Chat           | 53.3 |          33.3          |         10.0          |          66.7          |             80.0              |        66.7         |        63.3        |
| InternLM-20B-Chat            | 53.9 |          46.7          |         13.3          |          43.3          |             80.0              |        70.0         |        70.0        |
| ChatGPT                      | 55.6 |          46.7          |          6.7          |          86.7          |             83.3              |        63.3         |        46.7        |
| LLaMA-70B-Chat               | 57.2 |          63.3          |         20.0          |          53.3          |             80.0              |        66.7         |        60.0        |
| GPT-4                        | 81.1 |          93.3          |         36.7          |         100.0          |             90.0              |        83.3         |        83.3        |
| **Aquila2-34B-Chat**         | 58.3 |          43.3          |         16.7          |          63.6          |             80.0              |        80.0         |        66.7        |
| **Aquila2-34B-Chat+SFT**     | 65.6 |          73.3          |         16.7          |          76.7          |             80.0              |        76.7         |        70.0        |
| **Aquila2-34B-Chat+SFT+CoT** | 69.4 |          80.0          |         23.3          |          83.3          |             73.3              |        80.0         |        76.7        |

<br>

## Requirements

* python 3.10 and above
* pytorch 1.12 and above, 2.0 and above are recommended
* transformers 4.32 and above
* CUDA 11.4 and above are recommended (this is for GPU users, flash-attention users, etc.)
<br><br>

## Quickstart
We have provided a straightforward example to illustrate how to quickly get started with Aquila2.

Before proceeding, ensure that your environment is properly configured and that the necessary packages have been installed. First and foremost, ensure that these prerequisites are met and then follow the instructions below to install the necessary libraries and dependencies.
```
pip install -r requirements.txt
https://github.com/FlagAI-Open/FlagAI.git
(cd FlagAI/ && python setup.py install)
```


If your device supports fp16 or bf16 precision, we also recommend installing flash-attention to enhance execution speed and reduce memory consumption. It's important to note that flash-attention is optional, and the project can be executed normally without it.

For the installation of flash-attention, please refer to https://github.com/Dao-AILab/flash-attention/.

You can also set up the environment required for Aquila2 by directly[downloading the Docker file](https://model.baai.ac.cn/model-detail/220118) and installing it.

Now you can use <img src="assets/baai.png" width="14"/>  Modelhub or 🤗Transformers to run our model。

### <img src="assets/baai.png" width="18"/> ModelHub

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


# Example
test_data = [
    "Write a tongue twister that's extremely difficult to pronounce.",
]

for text in test_data:
    print(model.predict(text, tokenizer=tokenizer))
    # For Aquila2-7B or Aquila2-34B，you need to set sft=False
    # print(model.predict(text, tokenizer=tokenizer, sft=False))
```

The results of our execution are as follows:

```
Harry had a harpy flight, Fred had a fiddle, and George had a gecko for breakfast.  Say that three times fast and see how long you can make it last!
```

<br><br>



### 🤗 Transformers

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
device = torch.device("cuda")
model_info = "BAAI/AquilaChat2-7B"
tokenizer = AutoTokenizer.from_pretrained(model_info, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_info, trust_remote_code=True)
model.eval()
model.to(device)
text = "请给出10个要到北京旅游的理由。"
tokens = tokenizer.encode_plus(text)['input_ids'][:-1]
tokens = torch.tensor(tokens)[None,].to(device)
stop_tokens = ["###", "[UNK]", "</s>"]
with torch.no_grad():
    out = model.generate(tokens, do_sample=True, max_length=512, eos_token_id=100007, bad_words_ids=[[tokenizer.encode(token)[0] for token in stop_tokens]])[0]
    out = tokenizer.decode(out.cpu().numpy().tolist())
    print(out)
```


## Quantization

Before using quantization, BitsAndBytesConfig needs to be installed:

```
pip install bitsandbytes
```

After that, you're all set to use the quantized models for inference!


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

<br><br>

## Pretraining
### Usage
<br><br>

## Finetuning

### Usage
We provide users with a series of fine-tuning scripts designed to adapt models to various downstream tasks using custom data. Within the comments section of the scripts, users will find detailed instructions indicating which parameters may need adjustments based on specific needs.

Before initiating the fine-tuning process, you are required to have your training data prepared. All samples should be consolidated into a list and stored in a json file. Each sample should be represented as a dictionary, encompassing an ID and conversation, with the latter presented in list format. Below is an example for your reference:

```json
{
	"id": "alpaca_data.json_1",
	"conversations": [{
		"from": "human",
		"value": "What are the three primary colors?"
	}, {
		"from": "gpt",
		"value": "The three primary colors are red, blue, and yellow."
	}],
	"instruction": ""
}
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

<br><br>

## Demo

### Web UI
Our web will be coming soon.

<br><br>

## Long-Context Understanding

To extend the context length and break the bottleneck of training sequence length, we introduce several techniques, including NTK-aware interpolation, window attention, and LogN attention scaling.

<br><br>

## Tokenizer

Our tokenizer of BBPE type is trained on a 50GB corpus, mainly sampled from deduplicated Pile and deduplicated WuDao contents. We also add some special tokens for passage and conversation separation.
<br><br>

## Reproduction

For your reproduction of the model performance on benchmark datasets, we provide scripts for you to reproduce the results. Check [eval/README.md](eval/README.md) for more information. Note that the reproduction may lead to slight differences from our reported results.

<br><br>

## License Agreement



<br><br>

## Contact Us

If you are interested to leave a message to either our research team or product team, join our WeChat groups!

<img src="./assets/wechat-qrcode.jpg" width = "200" height = "200"  align=center />

