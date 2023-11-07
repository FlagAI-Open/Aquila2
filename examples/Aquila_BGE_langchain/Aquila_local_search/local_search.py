import sys
import json
import os
import shutil
import time
from typing import List
import gradio as gr
import nltk
import requests




from langchain.chains.base import Chain
from utils.model_config import *
from aqulia_server_chain import AquliaModel

sys.path.insert(0, "/mnt/share/baaisolution/zhaolulu/Aquila_BGE_langchain/BGE")

import argparse
from tool import SearchTool

LLM_MODEL = "aquilachat2-34b"

BGE_DATA_PATH = "/mnt/share/baaisolution/zhaolulu/Aquila_BGE_langchain/BGE/data/ai_filter.json"
BGE_ABSTRACT_EMB_PATH = "/mnt/share/baaisolution/zhaolulu/Aquila_BGE_langchain/BGE/data/abstract.npy"
BGE_ABSTRACT_INDEX_PATH = "/mnt/share/baaisolution/zhaolulu/Aquila_BGE_langchain/BGE/abstract.index"
BGE_ABSTRACT_BM25_INDEX_PATH = "/mnt/share/baaisolution/zhaolulu/Aquila_BGE_langchain/BGE/abstract_bm25_index"
BGE_META_EMB_PATH = "/mnt/share/baaisolution/zhaolulu/Aquila_BGE_langchain/BGE/data/meta.npy"
BGE_META_INDEX_PATH = "/mnt/share/baaisolution/zhaolulu/Aquila_BGE_langchain/BGE/meta.index"
BGE_META_BM25_INDEX_PATH = "/mnt/share/baaisolution/zhaolulu/Aquila_BGE_langchain/BGE/meta_bm25_index"
BGE_BATCH_SIZE = 128
BGE_SEARCH_NUM = 3

llm_model_dict = {
    "aquilachat2-34b": {
        "name": "aquilachat2-34b",
        "pretrained_model_name": "aquilachat2-34b",
        "local_model_path": "/mnt/share/baaisolution/zhaolulu/Aquila2/checkpoints",
        "provides": "Aquila"
    }
}

def generate_prompt(related_docs, query, prompt_template: str = PROMPT_TEMPLATE, ) -> str:
    context = []
    for idx, doc in related_docs.items():
        content = "The abstract is: " + doc["content"]["abstract"].replace("\n", " ") + " " + "The title is: " + doc["content"]["title"].replace(
            "\n", " ") + " " + "The author is: " + ",".join(doc["content"]["authors"]) + "."
        content = "source "+str(idx) + ": " + content

        context.append(content)
    context = "\n\n".join(context)

    prompt = prompt_template.replace("{question}", query).replace("{context}", context)
    return prompt


def torch_gc():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    elif torch.backends.mps.is_available():
        try:
            from torch.mps import empty_cache
            empty_cache()
        except Exception as e:
            print(e)
            print("如果您使用的是 macOS 建议将 pytorch 版本升级至 2.0.0 或更高版本，以支持及时清理 torch 产生的内存占用。")

class MyLocalDocQA():
    def __init__(self):
        self.llm_model_chain: Chain = None

    def init_cfg(self,llm_model: Chain = None):
        self.llm_model_chain = llm_model

    def get_search_result_from_BGE(self, query):
        search_tool = SearchTool(BGE_DATA_PATH,
                             BGE_ABSTRACT_EMB_PATH,
                             BGE_ABSTRACT_INDEX_PATH,
                             BGE_ABSTRACT_BM25_INDEX_PATH,
                             BGE_META_EMB_PATH,
                             BGE_META_INDEX_PATH,
                             BGE_META_BM25_INDEX_PATH,
                             128)

        input_text = query
        retrieval_type = "merge"
        query_type = "by query"
        target_type = "conditional"
        num = BGE_SEARCH_NUM
        rerank = "enable"
        rerank_num = 25
        response = search_tool.search(input_text, retrieval_type, query_type, target_type, num, rerank, rerank_num)
        return response

    def get_knowledge_based_answer_Aquila2_34b(self, query, chat_history):
        related_docs_with_score = self.get_search_result_from_BGE(query)
        torch_gc()
        if len(related_docs_with_score) > 0:
            prompt = generate_prompt(related_docs_with_score, query)
        else:
            prompt = query
        tokenizer = self.llm_model_chain.tokenizer
        answer = self.llm_model_chain.model.predict(prompt, tokenizer=tokenizer, model_name="aquilachat2-34b", max_gen_len=512)
        msg = ""
        for ans in answer.split():
            msg += ans
            msg += " "
            res = {"query": query,
                   "result": msg,
                   "source_documents": related_docs_with_score}
            chat_history[-1][0] = query
            if len(msg) % 5 == 0:
                time.sleep(0.25)
                yield res, chat_history
        yield res, chat_history
    

llm_model_dict_list = list(llm_model_dict.keys())

local_doc_qa = MyLocalDocQA()

flag_csv_logger = gr.CSVLogger()


def get_answer(query, history, mode):
    
    if mode == "文献库问答":
        logger.info("Use Aquila model to generate answer")
        for resp, res_history in local_doc_qa.get_knowledge_based_answer_Aquila2_34b(
                    query=query, chat_history=history + [[query, ""]]):
                source = f"{resp['result']}\n\n"
                #logger.info("source_documents is [{}]".format(resp["source_documents"]))
                source += "".join(
                    [f"""<details> <summary>出处 [{int(i) + 1}]:{doc["content"].get("title","")} </summary>\n"""
                     f"""{json.dumps(doc["content"],ensure_ascii=False)}\n"""
                     f"""</details>"""
                     for i, doc in resp["source_documents"].items()])

                res_history[-1][-1] = source
                yield res_history, ""

    logger.info(f"flagging: username={FLAG_USER_NAME},query={query},mode={mode},history={history}")
    flag_csv_logger.flag([query, history, mode], username=FLAG_USER_NAME)


def clear_history(request: gr.Request):
    return "", None


def init_model():
    if "aquilachat" in LLM_MODEL.lower():
        llm_model_ins = AquliaModel(llm_model_dict[LLM_MODEL])

    else:
        logger.info("加载模型不是Aquila系列，请手动添加需要加载的模型")

    try:
        local_doc_qa.init_cfg(llm_model=llm_model_ins)
        reply = """模型和数据已成功加载，可以开始对话查询，或从右侧选择模式后开始对话"""
        logger.info(reply)
      
        return reply
    except Exception as e:
        logger.error(e)
        reply = """模型未成功加载"""
        if str(e) == "Unknown platform: darwin":
            logger.info("该报错可能因为您使用的是 macOS 操作系统，需先下载模型至本地后执行 Web UI，具体方法请参考项目 README 中本地部署方法及常见问题："
                        " https://github.com/imClumsyPanda/langchain-ChatGLM")
        else:
            logger.info(reply)
        return reply




def change_mode(mode, history):
    if mode == "文献库问答":
        return gr.update(visible=True), gr.update(visible=False), history
    else:
        return gr.update(visible=False), gr.update(visible=False), history


block_css = """.importantButton {
    background: linear-gradient(45deg, #7e0570,#5d1c99, #6e00ff) !important;
    border: none !important;
}
.importantButton:hover {
    background: linear-gradient(45deg, #ff00e0,#8500ff, #6e00ff) !important;
    border: none !important;
}"""

webui_title = """
# 🎉 BAAI-LLM-SearchSystem !!! 🎉
👍 [实现代码点击链接跳转](https://gitee.com/BaaiAC/llm_app_model_service_hub.git)
"""
init_message = f""" 欢迎使用 BAAI-LLM-SearchSystem查询系统!!

支持本地文献库问答查询。

"""

# 初始化消息
model_status = init_model()

default_theme_args = dict(
    font=["Source Sans Pro", 'ui-sans-serif', 'system-ui', 'sans-serif'],
    font_mono=['IBM Plex Mono', 'ui-monospace', 'Consolas', 'monospace'],
)


with gr.Blocks(css=block_css, theme=gr.themes.Default(**default_theme_args)) as demo:
    file_status, model_status = gr.State(""), gr.State(model_status)
    gr.Markdown(webui_title)
    with gr.Tab("文献库检索"):
        with gr.Row():
            with gr.Column(scale=15):
                chatbot = gr.Chatbot([[None, init_message], [None, model_status.value]],
                                     elem_id="chat-box",
                                     show_label=False).style(height=600)
                query = gr.Textbox(show_label=False,
                                   placeholder="请输入提问内容，按回车进行提交").style(container=False)
                # 这里获取query
                clear_btn = gr.Button(value="新对话", interactive=True, elem_id="clear_btn")

            with gr.Column(scale=5):
                mode = gr.Radio(["文献库问答"],
                                label="请选择使用模式",
                                value="文献库问答", )

                examples = [
                    'find papers named "LSTM"',
                    'find papers on machine translation',
                    'can you give me one paper about CNN, show the title and authors?',
                    'find papers about "self attention", use all the abstract to generate a summary',
                    'give me some papers about Transformer and give one summary about these paper',
                    '能帮我找一篇光合作用相关的文章吗?'
                ]

                gr.Examples(examples, query, examples_per_page=20)

                flag_csv_logger.setup([query, chatbot, mode], "flagged")
                mode.change(fn=change_mode,
                            inputs=[mode, chatbot],
                            outputs=[chatbot])
                query.submit(get_answer,
                             [query, chatbot, mode],
                             [chatbot, query])
                clear_btn.click(clear_history, None, [chatbot, query])


(demo.queue(concurrency_count=3).launch(server_name='0.0.0.0',
                                        server_port=9172,
                                        show_api=False,
                                        share=False,
                                        inbrowser=False))
