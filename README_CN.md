<p align="left">
        中文</a>&nbsp ｜ &nbsp<a href="README.md">English</a>
</p>
<br><br>

<p align="center">
    <img src="./assets/logo.png" width="500"/>
<p>
<br>

<p align="center">
        🤗 <a href="https://huggingface.co/BAAI">Hugging Face</a>&nbsp&nbsp | &nbsp <a href="https://model.baai.ac.cn/models">ModelHub</a>&nbsp&nbsp | &nbsp&nbsp🖥️ <a href="https://modelscope.cn/studios/qwen/Qwen-14B-Chat-Demo/summary">Demo</a> | &nbsp&nbsp <a href="assets/wechat-qrcode.png">微信</a>
</p>
<br><br>

---介绍我们这次开源了哪些模型(7B/33B, base和chat)---

---加一个modelhub链接的表格---

---介绍一些Aquila2的优势---

---简单列一下接下来大纲---

---遇到问题的话怎么办，然后再放一波社群的链接---
<br><br>

## 更新

* 2023年10月x日，发布Aquila2 xxx版本

## 评测表现(袁野)

---介绍---

---多维图---

---表格---

<br><br>

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

### 对话模型推理

接下来可以使用`AquilaChat2-7B`对话模型来进行推理：

```
from flagai.auto_model.auto_loader import AutoLoader


# 模型名称
model_name = 'AquilaChat2-7B'

# 加载模型以及tokenizer
autoloader = AutoLoader("aquila2", model_name=model_name）
# 使用model_dir参数调整模型加载路径
# autoloader = AutoLoader("aquila2", model_dir='./checkpoints', model_name=model_name）
# 如需加载LoRA模型，需要额外提供LoRA模块的地址
# autoloader = AutoLoader("aquila2", model_name=model_name，lora_dir='./examples/checkpoints/lora/aquila2chat-hf'）
# 如需加载Q-LoRA模型，需要额外提供Q-LoRA模块的地址
# autoloader = AutoLoader("aquila2", model_name=model_name，qlora_dir='./examples/checkpoints/qlora/aquila2chat-hf'）

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
model in: A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.###Human: 北京的十大景点是什么?请将回答翻译成英文和日语###Assistant:
北京十大景点: 1. 天安门广场 2. 故宫 3. 颐和园 4. 天坛 5. 鸟巢 6. 北京大学 7. 清华大学 8. 北京动物园 9. 北京植物园 10. 长城。
model in: A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.###Human: 写一首中秋主题的五言绝句###Assistant:
皎洁月光洒九洲，团圆佳节倍思悠。
```

### 基础模型推理

基础模型推理与对话模型的不同在于模型推理的时候需要设置`sft=False`
```
from flagai.auto_model.auto_loader import AutoLoader


# 模型名称
model_name = 'AquilaChat2-7B'

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



## 量化

### 用法

---量化用法---

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

```
{"id": "alpaca_data.json_1", "conversations": [{"from": "human", "value": "What are the three primary colors?"}, {"from": "gpt", "value": "The three primary colors are red, blue, and yellow."}], "instruction": ""}
```

然后您可以使用我们提供不同的微调脚本实现不同功能：
- 使用`finetune/7B/finetune.sh`实现7B模型全参数微调 
- 使用`finetune/7B/finetune_lora.sh`实现7B模型LoRA微调 
- 使用`finetune/7B/finetune_qlora.sh`实现7B模型Q-LoRA微调 
- 使用`finetune/34B/finetune.sh`实现34B模型全参数微调 
- 使用`finetune/34B/finetune_lora.sh`实现34B模型LoRA微调 
- 使用`finetune/34B/finetune_qlora.sh`实现34B模型Q-LoRA微调 

注意，你需要在脚本中指定训练数据的路径, 并配置hostfile文件。若未在脚本里提供自定义的模型文件，脚本将会基于指定的模型名称自动从 ModelHub 下载相应的模型，并执行微调操作。


实现全参数微调只需运行如下脚本

```
# 微调7B模型
bash finetune/7B/finetune.sh
# 微调34B模型
bash finetune/34B/finetune.sh
```

LoRA (参见[论文](https://arxiv.org/abs/2106.09685)) 的微调方法与全参数微调有所不同。LoRA 仅更新 adapter 层的参数，而不更新原始语言模型的参数。这样做可以减小显存和计算开销，LoRA 适用于各种不同大小的模型和各种不同的任务，能够帮助用户更高效地微调模型以适应特定的任务或数据集。

实现LORA只需运行如下脚本
```
# 微调7B模型
bash finetune/7B/finetune_lora.sh
# 微调34B模型
bash finetune/34B/finetune_lora.sh
```

如果显存资源仍然受限，可以考虑使用 Q-LoRA (参见[论文](https://arxiv.org/abs/2305.14314))，这是一种通过使用4比特量化模型和 paged attention 技术，进一步降低显存使用的优化方案。

实现Q-LoRA只需运行如下脚本

```
# 微调7B模型
bash finetune/7B/finetune_qlora.sh
# 微调34B模型
bash finetune/34B/finetune_qlora.sh
```




### 优化效果

以下是7B和34B模型使用全参数微调，LoRA 和 QLoRA 处理不同输入长度时的显存占用和训练速度的数据。评测是在一台装备有 A100-SXM4-80G GPU 的机器上进行，使用 CUDA 12.1 和 Pytorch 2.1。其中7B模型的输入长度为2048， 34B模型的输入长度为4096。我们进行的所有测试均采用了批次大小为 4 和梯度累积为 1 的配置，并且记录了以GB为单位的显存占用和以s/iter为单位的训练速度。具体的数据如下：

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
        <th rowspan="2">34B</th><td>LoRA</td><td>LoRA</td><td>LoRA</td>
    </tr>
    <tr>
        <td>Q-LoRA</td><td>8.22s/it</td><td>37.7G</td>
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

FlagAI飞智大部分项目基于 [Apache 2.0 license](https://www.apache.org/licenses/LICENSE-2.0)
---可能还需要补充---
<br><br>

## 联系我们

* 官方邮箱：open.platform@baai.ac.cn。
* 知乎：[FlagAI飞智](https://www.zhihu.com/people/95-22-20-18)
* 扫码添加小助手加入**微信交流群**：

<img src="./assets/wechat-qrcode.jpg" width = "200" height = "200"  align=center />

