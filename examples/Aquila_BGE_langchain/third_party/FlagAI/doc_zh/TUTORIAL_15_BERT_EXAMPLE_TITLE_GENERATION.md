# BERT 标题生成例子

## 背景
标题生成任务需要输入一段文本，模型根据输入文本输出对应的标题。

![](./img/bert_title_generation_model.png)

## 结果展示

#### 输入
```
"本文总结了十个可穿戴产品的设计原则而这些原则同样也是笔者认为是这个行业最吸引人的地方1为人们解决重复性问题2从人开始而不是从机器开始3要引起注意但不要刻意4提升用户能力而不是取代人",
"2007年乔布斯向人们展示iPhone并宣称它将会改变世界还有人认为他在夸大其词然而在8年后以iPhone为代表的触屏智能手机已经席卷全球各个角落未来智能手机将会成为真正的个人电脑为人类发展做出更大的贡献",
"雅虎发布2014年第四季度财报并推出了免税方式剥离其持有的阿里巴巴集团15％股权的计划打算将这一价值约400亿美元的宝贵投资分配给股东截止发稿前雅虎股价上涨了大约7％至5145美元"
```
#### 输出
```
可 穿 戴 产 品 设 计 原 则 十 大 原 则
乔 布 斯 宣 布 iphone 8 年 后 将 成 为 个 人 电 脑
雅 虎 拟 剥 离 阿 里 巴 巴 15 ％ 股 权
```
## 使用

### 1. 数据加载
样例数据位于 /examples/bert_title_generation/data/

需要在 ```trainer.py```文件中定义数据读取过程，例如：
```python
def read_file():
    src = []
    tgt = []
    
    ## 如果换为其他数据，修改处理方式即可，只需要构造好src以及对应tgt列表
    with open(src_dir, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            src.append(line.strip('\n').lower())

    with open(tgt_dir, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            tgt.append(line.strip('\n').lower())
    return src,tgt
```

### 2. 加载模型与切词器

```python
from flagai.auto_model.auto_loader import AutoLoader

# model_dir: 包含 1.config.json, 2.pytorch_model.bin, 3.vocab.txt,
# 如果本地没有，则会在modelhub上进行查找并下载
# Autoloader 能够自动构建模型与切词器
# 'seq2seq' 是task_name
auto_loader = AutoLoader(task_name="title-generation",
                         model_dir="./state_dict/",
                         model_name="RoBERTa-base-ch")
model = auto_loader.get_model()
tokenizer = auto_loader.get_tokenizer()
```

### 3. 训练
在命令行中输入如下代码进行训练：
```commandline
python ./train.py
```
通过如下代码调整训练配置：
```python
from flagai.trainer import Trainer
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
trainer = Trainer(env_type="pytorch",
                  experiment_name="roberta_seq2seq",
                  batch_size=8, gradient_accumulation_steps=1,
                  lr = 2e-4,
                  weight_decay=1e-3,
                  epochs=10, log_interval=10, eval_interval=10000,
                  load_dir=None, pytorch_device=device,
                  save_dir="checkpoints",
                  save_interval=1
                  )
```
通过如下代码划分训练集验证集，并定义Dataset：
```python
sents_src, sents_tgt = read_file()
data_len = len(sents_tgt)
train_size = int(data_len * 0.8)
train_src = sents_src[: train_size]
train_tgt = sents_tgt[: train_size]

val_src = sents_src[train_size: ]
val_tgt = sents_tgt[train_size: ]

train_dataset = BertSeq2seqDataset(train_src, train_tgt, tokenizer=tokenizer, maxlen=maxlen)
val_dataset = BertSeq2seqDataset(val_src, val_tgt, tokenizer=tokenizer, maxlen=maxlen)
```

### 生成
如果你已经训练好了一个模型，为了更直观的看到结果，可以对一些测试样例进行生成
首先去修改一下模型保存位置：
```python
model_save_path = "./checkpoints/1001/mp_rank_00_model_states.pt" ## 1001 is example, you need modify the number.
```
运行对应的生成脚本：
```commandline
python ./generate.py
```
然后便可以看到生成结果。
