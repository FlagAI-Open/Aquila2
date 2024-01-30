# !/usr/bin/env python
# -*- coding:utf-8 -*-
# ==================================================================
# [Author]       : shixiaofeng
# [Descriptions] :
# ==================================================================
import uuid

# 基于上下文的prompt模版，请务必保留"{question}"和"{context}"
PROMPT_TEMPLATE = """The following information is known:
{context}

Based on the above known information, answer the user's questions concisely and professionally. If you can't get an answer from it, please say "the question cannot be answered based on known information" or "not enough relevant information is provided". Adding fabricated components to the answer is not allowed, and the answer should be in English. The question is：{question}"""

FLAG_USER_NAME = uuid.uuid4().hex

# query改写的数量
QUERY_REWRITE_NUM = 3

query_rewrite_prompt = """你是一个聪明的智能助手，可以根据单个输入查询语句生成多个与它相关的搜索查询语句。下面根据用户的问题生成{query_num}个用于查询的相关问题。要求逐行显示。

约定生成的回答格式为：
1. answer1
2. answer2
...
N. answerN

用户的问题为:{question}
生成回答:
"""


# 生成支持引文格式，中文格式，
# reference: https://arxiv.org/pdf/2305.14627.pdf
rag_prompt = {
    "reference_cont": "参考内容[{index}] (标题:{title}；摘要:{abstract}；作者:{authors})：{contents}",
    "inp_format": """你是一个专业，聪明的智能助手。使用提供的参考内容（其中一些可能无关，参考内容在两个'==='之间）编写用户给定问题的准确、简洁的答案，并确保正确引用。要使用无偏见的新闻语气。对任何事实主张都应提供引用。在引用多个搜索结果时，使用[1][2][3]。在每个句子中，至少引用1个文档，最多引用3个文档。如果多个文档支持这个句子，只引用最少的足够的文档子集。如果连续的句子都引用同一个文档，那么只在最后1个句子处引用文档。如果所有的参考内容都与给定的问题不相关，那么不依赖参考内容进行回答。

按照下面给出的两个例子进行生成

示例1开始
参考内容如下：
===
参考内容[0] (标题:提示信息；摘要:排放控制系统因发动机型号不同会有差异。；作者:无)：排放控制系统因发动机型号不同会有差异，具体配备请以实车为准。禁止对发动机或排放控制系统的任何部件进行改装。
参考内容[1] (标题:无；摘要:涡轮增压和自然吸气发动机的动力表现差异，以及昂科威和自由光在外观和尺寸上的差异。；作者:无)：涡轮PK自吸 有人觉得带“T”的发动机动力表现够强劲，再加上不大的排气量也能保证较低的油耗；又有人觉得涡轮增压发动机或多或少会在低转速时遇到动力迟滞，不如自然吸气发动机来的平顺。
===
参考内容全部结束。

用户给定的问题：排放控制系统是否会因为发动机型号不同而有差异？各有什么优势?
回答：排放控制系统会因发动机型号不同而有差异[0]。例如，涡轮增压发动机动力表现强劲，排气量小可以保证较低的油耗；但可能在低转速时遇到动力迟滞。相比之下，自然吸气发动机运行更平顺[1]。具体的设备配置需要以实车为准[0]。另外，对发动机或排放控制系统的任何部件进行改装是被禁止的[0]。
示例1结束

示例2开始
参考内容如下：
===
参考内容[0] (标题:提示信息；摘要:排放控制系统因发动机型号不同会有差异。；作者:无)：排放控制系统因发动机型号不同会有差异，具体配备请以实车为准。禁止对发动机或排放控制系统的任何部件进行改装。
参考内容[1] (标题:无；摘要:涡轮增压和自然吸气发动机的动力表现差异。；作者:无)：涡轮PK自吸 有人觉得带“T”的发动机动力表现够强劲，再加上不大的排气量也能保证较低的油耗；又有人觉得涡轮增压发动机或多或少会在低转速时遇到动力迟滞
===
参考内容全部结束。

燃油车有什么优势
回答：燃油车可能的优势包括更广泛的加油站网络、更长的驾驶距离（在需要加油/充电前）和更短的“加油”时间。然而，具体的优势可能取决于特定的车型和驾驶条件。
示例2结束

参考内容如下：
===
{refers}
===
参考内容全部结束。

用户给定的问题：{question}
回答：
""",
}


