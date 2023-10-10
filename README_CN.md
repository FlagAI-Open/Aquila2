<p align="left">
        中文</a>&nbsp ｜ &nbsp<a href="README.md">English</a>
</p>
<br><br>

<p align="center">
    <img src="./assets/logo.png" width="500"/>
<p>
<br>

<p align="center">
        🤗 <a href="https://huggingface.co/BAAI">Hugging Face</a>&nbsp&nbsp | &nbsp <a href="https://model.baai.ac.cn/models">ModelHub</a>&nbsp&nbsp | &nbsp&nbsp <a href="assets/wechat-qrcode.png">微信</a>
</p>
<br><br>

我们开源了我们的 **Aquila2** 系列，现在包括基础语言模型 **Aquila2-7B** 和 **Aquila2-34B** ，以及对话模型 **Aquila2-7B-Chat** 和 **Aquila2-34B-Chat**。

| 模型名称         | Modelhub  | Huggingface | 
|----------------------|:----:|:-----------: |
| Aquila2-7B | https://model.baai.ac.cn/model-detail/100118 |    -     | 
| AquilaChat2-7B | https://model.baai.ac.cn/model-detail/100117 |   -      | 
| Aquila2-34B | https://model.baai.ac.cn/model-detail/100119  |    -    | 
| AquilaChat2-34B | https://model.baai.ac.cn/model-detail/100116 |   -      |

在这个仓库中，您可以：

* 快速开始使用 Aquila2，进行简单的推理。
* 有关量化模型的详细信息，包括使用方法、内存、推理速度。为了比较，我们还提供了 BF16 模型的统计数据。
* 微调教程，包括全参数调优、LoRA 和 Q-LoRA。
* 长文本理解评估的统计数据
* 许可协议
* ...

欢迎对我们提出任何问题（建议用英语，这样更多人会明白你的问题哦）！如果有兴趣帮我们改进 **Aquila2**，可以提交你的Pull Requests， 我们会及时处理。

如果你想与我们进行讨论和交流，请尽快加入我们的微信群吧(请参见文档顶部以获取入口信息)！



<br>

## 更新

* 2023.10.10 🔥 我们在 ModelHub 和 Hugging Face 上发布了 **Aquila2-34B** 和 **Aquila2-34B-Chat**。

## 评测表现

Aquila2-34B和Aquila2-7B（最新版本使用了更多数据和更长的上下文进行了训练，上下文长度从2048扩展到了8192）相比同规模的基线模型在各项评测数据集上均表现更优，评测数据集包括MMLU、C-Eval、GSM8K、MATH、HumanEval等，考察了模型的自然语言理解能力、数学问题求解能力、代码能力等各方面能力。

### 基础模型表现

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

### 对话模型表现

|      Model          | Placeholder |
| :-----------------: | :---------: |
| **AquilaChat2-7B**  |             |
| **AquilaChat2-33B** |             |

<br>

### 长文本任务表现
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

### 推理任务表现

| Model                        | bAbI#16<br>(Inductive) | CLUTRR<br>(Inductive) | bAbI#15<br>(Deductive) | EntailmentBank<br>(Deductive) | αNLI<br>(Abductive) | E-Care<br>(Casual) | Avg. |
| :--------------------------- | :--------------------: | :-------------------: | :--------------------: | :---------------------------: | :-----------------: | :----------------: | :--: |
| Baichuan2-7B-Chat            |          40.0          |         26.7          |          43.3          |             73.3              |        53.3         |        50.0        | 47.8 |
| Qwen-7B-Chat                 |          20.0          |         10.0          |          66.7          |             86.7              |        56.7         |        56.7        | 49.5 |
| Qwen-14B-Chat                |          26.7          |         10.0          |          63.3          |             86.7              |        63.3         |        56.7        | 51.1 |
| Baichuan2-13B-Chat           |          33.3          |         10.0          |          66.7          |             80.0              |        66.7         |        63.3        | 53.3 |
| InternLM-20B-Chat            |          46.7          |         13.3          |          43.3          |             80.0              |        70.0         |        70.0        | 53.9 |
| ChatGPT                      |          46.7          |          6.7          |          86.7          |             83.3              |        63.3         |        46.7        | 55.6 |
| LLaMA-70B-Chat               |          63.3          |         20.0          |          53.3          |             80.0              |        66.7         |        60.0        | 57.2 |
| GPT-4                        |          93.3          |         36.7          |         100.0          |             90.0              |        83.3         |        83.3        | 81.1 |
| **Aquila2-34B-Chat**         |          43.3          |         16.7          |          63.6          |             80.0              |        80.0         |        66.7        | 58.3 |
| **Aquila2-34B-Chat+SFT**     |          73.3          |         16.7          |          76.7          |             80.0              |        76.7         |        70.0        | 65.6 |
| **Aquila2-34B-Chat+SFT+CoT** |          80.0          |         23.3          |          83.3          |             73.3              |        80.0         |        76.7        | 69.4 |

<br>

## 安装环境

* Python 版本 >= 3.8
* PyTorch 版本 >= 1.8.0
* CUDA 版本 >= 11.7（GPU用户、flash-attention用户等需考虑此选项）
<br>

## 快速使用

我们为您展示了一个简单的示例, 来演示如何快速上手Aquila2.

在您动手操作之前，请确认您已经设置好了运行环境，并成功安装了必要的代码包。首先，请确保满足这些先决条件，然后按照下面的指示安装必要的库和依赖。

```
pip install -r requirements.txt
```

如果您的显卡兼容 fp16 或 bf16 精度，我们还建议您安装 flash-attention，以增加运行速度和减少显存使用。请注意，flash-attention 不是必须的，没有它您也能正常执行该项目。

flash-attention安装：参考 https://github.com/Dao-AILab/flash-attention/

现在可以开始使用 Transformers 或 Modelhub 来运行我们的模型。


### ModelHub

要使用 Aquila2-Chat 进行推理，你只需要输入下面演示的几行代码。

```python
from flagai.auto_model.auto_loader import AutoLoader


# 模型名称
model_name = 'AquilaChat2-7B'
# model_name = 'AquilaChat2-34B'

# 加载模型以及tokenizer
autoloader = AutoLoader("aquila2", model_name=model_name)
# 使用model_dir参数调整模型加载路径
# autoloader = AutoLoader("aquila2", model_dir='./checkpoints', model_name=model_name)
# 如需加载LoRA模块，需要额外提供LoRA模块的地址
# autoloader = AutoLoader("aquila2", model_name=model_name，lora_dir='./examples/checkpoints/lora/aquila2chat-hf')
# 如需加载Q-LoRA模块，需要额外提供Q-LoRA模块的地址
# autoloader = AutoLoader("aquila2", model_name=model_name，qlora_dir='./examples/checkpoints/qlora/aquila2chat-hf')

model = autoloader.get_model()
tokenizer = autoloader.get_tokenizer()


# 对话测试样例
test_data = [
    "北京的十大景点是什么?请将回答翻译成英文和日语",
    "写一首中秋主题的五言绝句",
]

for text in test_data:
    print(model.predict(text, tokenizer=tokenizer))
```

我们运行的结果如下:
```
北京十大景点: 1. 天安门广场 2. 故宫 3. 颐和园 4. 天坛 5. 鸟巢 6. 北京大学 7. 清华大学 8. 北京动物园 9. 北京植物园 10. 长城。

皎洁月光洒九洲，团圆佳节倍思悠。
```

基础模型推理的用法类似，与对话模型的不同之处只在于模型推理的时候需要设置`sft=False`

<details>
  <summary>Aquila2基础模型推理</summary>

```python
from flagai.auto_model.auto_loader import AutoLoader


# 模型名称
model_name = 'Aquila2-7B'
# model_name = 'Aquila2-34B'

# 加载模型以及tokenizer
autoloader = AutoLoader("aquila2", model_name=model_name)

model = autoloader.get_model()
tokenizer = autoloader.get_tokenizer()

# 对话测试样例
test_data = [
    "北京的十大景点是什么?请将回答翻译成英文和日语",
    "写一首中秋主题的五言绝句",
]

for text in test_data:
    print(model.predict(text, tokenizer=tokenizer, sft=False))
```

</details>


## 量化

### 用法

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
    "北京的十大景点是什么?请将回答翻译成英文和日语",
    "写一首中秋主题的五言绝句",
    "Write a tongue twister that's extremely difficult to pronounce.",
]

for text in test_data:
    print(model.predict(text, tokenizer=tokenizer))

```

### 效果评测

---量化效果(可选)---

### 推理速度

---量化推理速度(可选)---

### 显存使用

---量化显存使用(可选)---
<br><br>

## 微调

我们为用户提供了一系列微调脚本，用于在自定义数据上微调模型，以适应不同的下游任务。在脚本的注释部分，用户会找到详细的说明，指明哪些参数需要根据实际需求进行调整。

在进行微调操作之前，您必须先准备好您的训练数据。所有样本需要集中到一个列表中，并存储在一个 json 文件里。每个样本应表现为一个字典，包括 id 和 conversation，其中，conversation 以列表的形式展现。以下提供了一个示例：

```json
{"id": "alpaca_data.json_1", "conversations": [{"from": "human", "value": "What are the three primary colors?"}, {"from": "gpt", "value": "The three primary colors are red, blue, and yellow."}], "instruction": ""}
```

然后您可以使用我们提供不同的微调脚本实现不同功能：
- 使用`finetune/7B/finetune.sh`实现7B模型全参数微调 
- 使用`finetune/7B/finetune_lora.sh`实现7B模型LoRA微调 
- 使用`finetune/7B/finetune_qlora.sh`实现7B模型Q-LoRA微调 
- 使用`finetune/34B/finetune.sh`实现34B模型全参数微调 
- 使用`finetune/34B/finetune_lora.sh`实现34B模型LoRA微调 
- 使用`finetune/34B/finetune_qlora.sh`实现34B模型Q-LoRA微调 

Note that you are required to specify the path to the training data within the script, and configure the hostfile accordingly. If a custom model file is not provided in the script, it will automatically download the corresponding model from ModelHub based on the specified model name and proceed with the fine-tuning operation.


To perform full-parameter fine-tuning, execute the following scripts:

```bash
# Fine-tuning the 7B model
bash finetune/7B/finetune.sh
# Fine-tuning the 34B model
bash finetune/34B/finetune.sh
```

The fine-tuning approach of LoRA (as detailed in the [paper](https://arxiv.org/abs/2106.09685)) varies from the full-parameter method. LoRA solely updates the parameters of the adapter layer without modifying the original language model parameters. This practice reduces memory and computational overhead. Applicable to a variety of model sizes and tasks, LoRA facilitates more efficient model fine-tuning to cater to specific tasks or datasets.

To implement LoRA, execute the following scripts:

```bash
# 微调7B模型
bash finetune/7B/finetune_lora.sh
# 微调34B模型
bash finetune/34B/finetune_lora.sh
```

If memory resources remain constrained, consider employing Q-LoRA (refer to the [paper](https://arxiv.org/abs/2305.14314)), an optimized solution that further reduces memory usage through the utilization of 4-bit quantized models and paged attention techniques.

To implement Q-LoRA, execute the following scripts:

```bash
# 微调7B模型
bash finetune/7B/finetune_qlora.sh
# 微调34B模型
bash finetune/34B/finetune_qlora.sh
```




### 优化效果

以下是7B和34B模型使用全参数微调，LoRA 和 QLoRA 处理不同输入长度时的显存占用和训练速度的数据。评测是在一台装备有 A100-SXM4-80G GPU 的机器上进行，使用 CUDA 12.1 和 Pytorch 2.1。其中7B模型的输入长度为2048， 34B模型的输入长度为4096。我们进行的所有测试均采用了批次大小为 4 和梯度累积为 1 的配置，并且记录了以GB为单位的显存占用和以s/iter为单位的训练速度。具体的数据如下：

<table>
    <tr>
      <th>模型大小</th><th>微调方法</th><th>显存占用</th><th>训练速度</th>
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

## 预训练

---预训练使用(玉龙)---
<br><br>

## 长文本理解

---介绍---

---评测结果---

## Tokenization

---中文可以简单说说tokenization是什么（因为这词没有好的中文对应翻译）---

---给一个tokenizer文档的link(可选)---
<br><br>

## 复现

---复现评测的脚本(可选)---
<br><br>

## FAQ

欢迎在 [GitHub Issues](https://github.com/FlagAI-Open/FlagAI/issues) 中提出你的问题，或在 [Discussions ](https://github.com/FlagAI-Open/FlagAI/discussions) 板块交流使用经验。

---之后可以弄一个常见问题的文档link放到这里---
<br><br>

## 使用协议

Aquila2项目基于 [Apache 2.0 license](https://www.apache.org/licenses/LICENSE-2.0)
---可能还需要补充---
<br><br>

## 联系我们

* 官方邮箱：open.platform@baai.ac.cn。
* 知乎：[FlagAI飞智](https://www.zhihu.com/people/95-22-20-18)
* 扫码添加小助手加入**微信交流群**：

<img src="./assets/wechat-qrcode.jpg" width = "200" height = "200"  align=center />

