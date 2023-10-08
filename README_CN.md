<p align="left">
        中文</a>&nbsp ｜ &nbsp<a href="README.md">English</a>
</p>
<br><br>

<p align="center">
    <img src="./assets/logo.png" width="500"/>
<p>
<br>

<p align="center">
        🤗 <a href="https://huggingface.co/BAAI/AquilaChat-7B">Hugging Face</a>&nbsp&nbsp | &nbsp <a href="https://model.baai.ac.cn/models">ModelHub</a>&nbsp&nbsp | &nbsp&nbsp🖥️ <a href="https://modelscope.cn/studios/qwen/Qwen-14B-Chat-Demo/summary">Demo</a> | &nbsp&nbsp <a href="assets/wechat-qrcode.png">微信</a>
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

<!-- <p align="left">
    <img src="assets/radar_14b.jpg" width="600"/>
<p>
<br>

| Model                  |   MMLU   |  C-Eval  |  GSM8K   |   MATH   | HumanEval |   MBPP    |   BBH    |  CMMLU   |
|:-----------------------|:--------:|:--------:|:--------:|:--------:|:---------:|:---------:|:--------:|:--------:|
|                        |  5-shot  |  5-shot  |  8-shot  |  4-shot  |  0-shot   |  3-shot   |  3-shot  |  5-shot  |
| LLaMA2-7B              |   46.8   |   32.5   |   16.7   |   3.3    |   12.8    |   20.8    |   38.2   |   31.8   |
| LLaMA2-13B             |   55.0   |   41.4   |   29.6   |   5.0    |   18.9    |   30.3    |   45.6   |   38.4   |
| LLaMA2-34B             |   62.6   |    -     |   42.2   |   6.2    |   22.6    |   33.0    |   44.1   |    -     |
| ChatGLM2-6B            |   47.9   |   51.7   |   32.4   |   6.5    |     -     |     -     |   33.7   |    -     |
| InternLM-7B            |   51.0   |   53.4   |   31.2   |   6.3    |   10.4    |   14.0    |   37.0   |   51.8   |
| InternLM-20B           |   62.1   |   58.8   |   52.6   |   7.9    |   25.6    |   35.6    |   52.5   |   59.0   |
| Baichuan2-7B           |   54.7   |   56.3   |   24.6   |   5.6    |   18.3    |   24.2    |   41.6   |   57.1   |
| Baichuan2-13B          |   59.5   |   59.0   |   52.8   |   10.1   |   17.1    |   30.2    |   49.0   |   62.0   |
| **Qwen-7B (original)** |   56.7   |   59.6   |   51.6   |     10.4     |   24.4    |   31.2    |   40.6   |   58.8   |
| **Qwen-7B**            |   58.2   |   63.5   |   51.7   |   11.6   |   29.9    |   31.6    |   45.0   |   62.2   |
| **Qwen-14B**           | **66.3** | **72.1** | **61.3** | **24.8** | **32.3**  | **40.8**  | **53.4** | **71.0** |


对于以上所有对比模型，我们列出了其官方汇报结果与[OpenCompass](https://opencompass.org.cn/leaderboard-llm)结果之间的最佳分数。

更多的实验结果和细节请查看我们的技术备忘录。点击[这里](https://qianwen-res.oss-cn-beijing.aliyuncs.com/QWEN_TECHNICAL_REPORT.pdf)。 -->
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

接下来可以使用Aquila2模型来进行推理：

```
from flagai.auto_model.auto_loader import AutoLoader

# 模型位置
state_dict = "./checkpoints/"

# 模型名称
model_name = 'Aquila2Chat-hf'

# 加载模型以及tokenizer
autoloader = AutoLoader("aquila2", model_dir=state_dict, model_name=model_name）
# 使用torch_dtype参数调整模型精度
# autoloader = AutoLoader("aquila2", model_dir=state_dict, model_name=model_name，torch_dtype=torch.bfloat16）
# 如需加载lora模型，需要额外提供lora模块的地址
# autoloader = AutoLoader("aquila2", model_dir=state_dict, model_name=model_name，lora_dir='./examples/checkpoints/lora/aquila2chat-hf'）
# 如需加载lora模型，需要额外提供qlora模块的地址
# autoloader = AutoLoader("aquila2", model_dir=state_dict, model_name=model_name，qlora_dir='./examples/checkpoints/qlora/aquila2chat-hf'）

model = autoloader.get_model()
tokenizer = autoloader.get_tokenizer()


# 对话测试样例
test_data = [
    "北京的十大景点是什么?请将回答翻译成英文和日语",
    "写一首中秋主题的五言绝句并翻译成英文和韩语",
]

for text in test_data:
    print(model.predict(text, tokenizer=tokenizer))
```

---加一个transformers的用法---


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

备好数据后，你可以使用我们提供的shell脚本实现微调。注意，你需要在脚本中指定你的数据的路径。

若未提供自定义的模型文件，脚本将会基于指定的模型名称自动从 ModelHub 下载相应的模型，并执行微调操作。

先进入`./examples`目录
```
cd examples
```

然后您可以使用不同的微调脚本实现不同功能：
- 使用`./finetune.sh`实现全参数微调 
- 使用`./finetune_lora.sh`实现LoRA微调 
- 使用`./finetune_qlora.sh`实现Q-LoRA微调 



实现全参数微调只需运行如下脚本

```
bash finetune.sh

```

LoRA (参见[论文](https://arxiv.org/abs/2106.09685)) 的微调方法与全参数微调有所不同。LoRA 仅更新 adapter 层的参数，而不更新原始语言模型的参数。这样做可以减小显存和计算开销，LoRA 适用于各种不同大小的模型和各种不同的任务，能够帮助用户更高效地微调模型以适应特定的任务或数据集。

实现LORA只需运行如下脚本
```
bash finetune_lora.sh
```

如果显存资源仍然受限，可以考虑使用 Q-LoRA (参见[论文](https://arxiv.org/abs/2305.14314))，这是一种通过使用4比特量化模型和 paged attention 技术，进一步降低显存使用的优化方案。

实现Q-LoRA只需运行如下脚本

```
bash finetune_qlora.sh
```

### 调整参数
下面是微调过程中一些可调整的重要参数:

|   参数名          |  类型   | 描述  |                                  
| :---------------- | :------- | :-- |   
| per_device_train_batch_size         | int  |   每次迭代训练时，从数据集中抽取的样本数。一般来说，它越大，处理速度越快，但会占用更多的内存。   |
| gradient_accumulation_steps          |int  |    在更新模型权重之前，要对多个小批次进行梯度计算的次数。主要应用于GPU显存较小的情况下，可以使用小的batch_size，通过梯度累积达到与大batch_size相同的效果。    |
| learning_rate          |float  |    指控制模型更新参数时的步长或速率。学习率过高可能导致模型不收敛，而学习率过低则可能导致训练时间过长或者陷入局部最优解。   |  
| gradient_checkpointing           |bool |    一种内存优化技术，用于减少神经网络训练过程中的 GPU 或其他计算设备的内存使用量。这种技术特别有用对于那些有限的硬件资源，但需要训练大型神经网络的情况。 | 
| warmup_ratio           |float |   初始学习率与原始学习率的比例。     | 
| save_strategy          | str  |    保存模型权重的策略，当训练时间较长时，保存间隔可以避免因突然中断或出现错误导致训练成果全部丢失; 可选项有: 1.'epoch'代表在每一轮训练结束时保存权重 2. 'steps'代表每隔一定步数保存一次模型，具体的步数在`save_steps`参数里指定。   |   
| logging_steps           |int  |    日志输出的间隔，即每训练多少个iteration输出一次日志信息。    | 
| use_lora           |bool  |    是否启用lora微调。   | 
| q_lora           |bool  |    是否启用qlora微调, 需要`use_lora`为true时，此参数才会生效。   | 
| lora_r          |int  |    `lora_r`是低秩适应的秩。这个参数控制了低秩适应的复杂性。通过调整`lora_r`的值，可以控制降维的程度，从而影响模型的性能和效率。较小的 lora_r 值可能导致更简单、更快的模型，但可能牺牲一些性能。    |  
| lora_alpha           |int  |    在 LoRA 中，`lora_alpha`和`lora_r`的比率通常用来调整低秩适应层的学习率。具体来说，该比率将决定低秩适应层的学习率相对于原始模型其他部分的学习率的倍数。通过调整这个比率，你可以控制 LoRA 层参数更新的速度。   | 
| lora_dropout           |float |    `lora_dropout`是dropout率。在深度学习和神经网络中，dropout是一种正则化技术，通过随机关闭一部分神经元来防止过拟合。   | 



### 优化效果

7B 全参, 2048: 2.67s/it, 43.9G
lora: 2.04s/it, 29.4G
qlora: 2.14s/it, 19.9G

34B, qlora, 37.7G, 8.22s/it



<table>
    <tr>
      <th rowspan="2">Model Size</th><th rowspan="2">Method</th><th colspan="4" align="center">Sequence Length</th>
    </tr>
    <tr>
        <th align="center">256</th><th align="center">512</th><th align="center">1024</th><th align="center">2048</th>
    </tr>
    <tr>
        <th rowspan="2">7B</th><td>LoRA</td><td align="center">33.5G / 1.6s/it</td><td align="center">34.0G / 1.7s/it</td><td align="center">35.0G / 3.0s/it</td><td align="center">35.0G / 5.7s/it</td>
    </tr>
    <tr>
        <td>Q-LoRA</td><td align="center">11.5G / 3.0s/it</td><td align="center">12.2G / 3.6s/it</td><td align="center">12.7G / 4.8s/it</td><td align="center">13.9G / 7.3s/it</td>
    </tr>
    <tr>
        <th rowspan="2">14B</th><td>LoRA</td><td align="center">51.0G / 2.1s/it</td><td align="center">51.0G / 2.7s/it</td><td align="center">51.5G / 5.0s/it</td><td align="center">53.9G / 9.2s/it</td>
    </tr>
    <tr>
        <td>Q-LoRA</td><td align="center">18.3G / 5.4s/it</td><td align="center">18.4G / 6.4s/it</td><td align="center">18.5G / 8.5s/it</td><td align="center">19.9G / 12.4s/it</td>
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

