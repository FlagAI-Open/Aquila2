# !/usr/bin/env python
# -*- coding:utf-8 -*-
# ==================================================================
# [CreatedDate]  : Thursday, 1970-01-01 08:00:00
# [Author]       : shixiaofeng
# [Descriptions] :
# ==================================================================
# [ChangeLog]:
# [Date]    	[Author]	[Comments]
# ------------------------------------------------------------------
import json
import time
from typing import List
import gradio as gr
import requests
import copy
import traceback
from utils.model_config import *
from utils.prompt_temp import *


def torch_gc():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    elif torch.backends.mps.is_available():
        try:
            from torch.mps import empty_cache

            empty_cache()
        except Exception as e:
            logger.info(e)
            logger.info(
                "如果您使用的是 macOS 建议将 pytorch 版本升级至 2.0.0 或更高版本，以支持及时清理 torch 产生的内存占用。"
            )


def request_to_llm(prompt, stream=False, llm_model=None):
    if LLM_URL:
        data = copy.deepcopy(LLM_CONFIG)
        data.update({"prompt": prompt})
        inp_data = json.dumps(data)
        logger.info(f"request_to_llm inp_data: {inp_data}")
        if not stream:
            response = requests.post(
                LLM_URL + "/batch_func", json=inp_data, stream=False
            )
            result = json.loads(response.text)["completions"][0]["text"]
            return result
        else:
            response = requests.post(
                LLM_URL + "/stream_func", json=inp_data, stream=True
            )
            print("下面是流式输出内容")
            return response.iter_content(chunk_size=None)
    else:
        assert llm_model is not None
        # 非stream方式
        if not stream:
            result = llm_model.model.predict(
                prompt,
                tokenizer=llm_model.tokenizer,
                model_name=LLM_MODEL_NAME,
                max_gen_len=LLM_CONFIG.get("max_gen_len", 512),
                convo_template=LLM_CONFIG.get("template", "aquila-v2"),
                seed=LLM_CONFIG.get("seed", 1234),
                topk=LLM_CONFIG.get("top_k_per_token", 15),
                top_p=LLM_CONFIG.get("top_p", 0.9),
                sft=LLM_CONFIG.get("sft", True),
            )
        ## stream
        else:
            inp_config = {
                "prompt": prompt,
                "top_k_per_token": LLM_CONFIG.get("top_k_per_token", 15),
                "top_p": LLM_CONFIG.get("top_p", 0.9),
                "seed": LLM_CONFIG.get("seed", 1234),
                "time": LLM_CONFIG.get("gene_time", 15),
                "temperature": LLM_CONFIG.get("temperature", 1.0),
                "template": LLM_CONFIG.get("template", "aquila-v2"),
                "max_new_tokens": LLM_CONFIG.get("max_new_tokens", 512),
                "sft": LLM_CONFIG.get("sft", True),
            }

            result = llm_model.gen_stream(inp_config)

        return result


def request_to_bge(one_query, url="", search_tool=None):
    # 访问bge得到模型返回结果
    inp_data = copy.deepcopy(retrival_configs)
    inp_data.update({"query": one_query})
    if SearchToolURL:
        response = requests.post(SearchToolURL + "/search", json=inp_data, stream=False)
        data = json.loads(response.text)
        return data
    else:
        assert SearchToolClient is not None
        response = SearchToolClient.search(**inp_data)
        return response


# RRF: Reciprocal Rank Fusion algorithm
def reciprocal_rank_fusion(search_results_dict, k=60):
    fused_scores = {}
    logger.info("Initial individual search result ranks:")
    for query, doc_scores in search_results_dict.items():
        logger.info(f"For query '{query}': {doc_scores}")

    for query, doc_scores in search_results_dict.items():
        for rank, (_, doc) in enumerate(
            sorted(
                doc_scores.items(), key=lambda kv: kv[1]["rerank score"], reverse=True
            )
        ):
            doc = json.dumps(doc, ensure_ascii=False)
            if doc not in fused_scores:
                fused_scores[doc] = 0
            previous_score = fused_scores[doc]
            fused_scores[doc] += 1 / (rank + k)
            logger.info(
                f"Updating score for {doc} from {previous_score} to {fused_scores[doc]} based on rank {rank} in query '{query}'"
            )

    reranked_results = [
        [json.loads(doc), score]
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]

    logger.info("Final reranked results:", reranked_results)

    return reranked_results


def query_fusion_retrival(user_query, llm_model=None):
    ## 流程:
    # 1. query改写，生成多个相关query
    # 2. 多个query并行召回
    # 3. RRF倒数排排序融合
    # 4. 截取topk相关doc
    logger.info("query_fusion_retrival")
    try:
        prompt = query_rewrite_prompt.format(
            query_num=QUERY_REWRITE_NUM, question=user_query
        )
        logger.debug(f"query_rewrite_prompt is [{prompt}]")
        response = request_to_llm(prompt, llm_model=llm_model)
        question_spl = response.strip().split("\n")
        rewrite_query = [
            one.split(".")[-1].replace(" ", "") for one in question_spl if one
        ]
        rewrite_query = [user_query] + rewrite_query
    except:
        rewrite_query = [user_query]
        traceback.print_exc()
    all_results = {}
    for query in rewrite_query:
        try:
            search_results = request_to_bge(query)
            all_results[query] = search_results
        except:
            all_results[query] = []

    reranked_results = reciprocal_rank_fusion(all_results)
    reranked_results = reranked_results[:BGE_TOPK_NUM]
    fused_docs = {}
    for index, (doc, score) in enumerate(reranked_results):
        doc["fused_score"] = score
        fused_docs[index] = doc
    logger.info(f"fused_docs: {fused_docs}")
    return fused_docs


def generate_rag_prompt(user_query, docs):
    all_refs = []
    for index, value in docs.items():
        content = value["content"]
        ref = rag_prompt["reference_cont"].format(
            index=index,
            title=content["title"] if content["title"] else "无",
            abstract=content["abstract"] if content["abstract"] else "无",
            authors=content["authors"] if content["authors"] else "无",
            contents=content["details"].replace("\n", "").strip()
            if content.get("details", "")
            else "无",
        )
        all_refs.append(ref)
    processed_prompt = rag_prompt["inp_format"].format(
        refers="\n".join(all_refs), question=user_query
    )

    return processed_prompt


class RagFusionPipe:
    def __init__(self):
        self.llm_model = None

    def init_cfg(self, llm_model):
        self.llm_model = llm_model

    def get_retrieval_result(self, query):
        from models.bge.retrival_client import request_for_rag_info

        response = request_for_rag_info(query)
        return response

    def forward(self, query, chat_history):
        logger.info("request come in")
        fused_query_retrival_docs = query_fusion_retrival(
            query, llm_model=self.llm_model
        )
        #
        processed_prompt = generate_rag_prompt(query, fused_query_retrival_docs)
        # 目前关闭了history功能
        response_iter = request_to_llm(
            processed_prompt, stream=True, llm_model=self.llm_model
        )
        msg = ""
        for ans in response_iter:
            if isinstance(ans, bytes):
                ans = ans.decode("utf8")
            msg += ans
            # msg += " "
            res = {
                "query": query,
                "result": msg,
                "source_documents": fused_query_retrival_docs,
            }
            chat_history[-1][0] = query
            yield res, chat_history
        yield res, chat_history

    def get_knowledge_based_answer_Aquila2_34b(self, query, chat_history):
        related_docs_with_score = self.get_retrieval_result(query)
        torch_gc()
        if len(related_docs_with_score) > 0:
            prompt = generate_prompt(related_docs_with_score, query)
        else:
            prompt = query
        tokenizer = self.llm_model.tokenizer
        answer = self.llm_model.model.predict(
            prompt, tokenizer=tokenizer, model_name="aquilachat2-34b", max_gen_len=512
        )
        msg = ""
        for ans in answer.split():
            msg += ans
            msg += " "
            res = {
                "query": query,
                "result": msg,
                "source_documents": related_docs_with_score,
            }
            chat_history[-1][0] = query
            if len(msg) % 5 == 0:
                time.sleep(0.25)
                yield res, chat_history
        yield res, chat_history


llm_model_dict_list = list(llm_model_dict.keys())

rag_fusion_pipe = RagFusionPipe()

flag_csv_logger = gr.CSVLogger()


def get_answer(query, history, mode):
    if mode == "知识库问答":
        logger.info("Use Aquila model to generate answer")
        for resp, res_history in rag_fusion_pipe.forward(
            query=query, chat_history=history + [[query, ""]]
        ):
            source = f"{resp['result']}\n\n"
            # 参考内容折叠起来
            source += "".join(
                [
                    f"""<details> <summary>出处 [{int(i)}]:{doc["content"].get("title","")[:30]} - {doc["content"].get("abstract","")[:30]} </summary>\n"""
                    f"""{json.dumps(doc["content"],ensure_ascii=False)}\n"""
                    f"""</details>"""
                    for i, doc in resp["source_documents"].items()
                ]
            )
            res_history[-1][-1] = source
            yield res_history, ""

    logger.info(
        f"flagging: username={FLAG_USER_NAME},query={query},mode={mode},history={history}"
    )
    flag_csv_logger.flag([query, history, mode], username=FLAG_USER_NAME)


def clear_history(request: gr.Request):
    return "", None


def init_model():
    ## NOTE: 有LLM_MODEL_NAME 并且没有提供LLM_URL
    if "aquilachat" in LLM_MODEL_NAME.lower() and LLM_URL == "":
        FlagAI_PATH = "/share/project/shixiaofeng/code/FlagAI-official"
        print(f"请提供FlagAI的目录: {FlagAI_PATH}")
        sys.path.insert(0, FlagAI_PATH)
        from models.aquila.aquila_model_local import AquliaModel

        llm_model = AquliaModel(llm_model_dict[LLM_MODEL_NAME])
    else:
        logger.info(
            f"加载模型不是Aquila系列或者不需要本地加载，请确认模型名{LLM_MODEL_NAME.lower()}是否正确，或提供的url是否正确{LLM_URL}"
        )
        llm_model = None
    try:
        rag_fusion_pipe.init_cfg(llm_model=llm_model)
        reply = """模型和数据已成功加载，可以开始对话查询，或从右侧选择模式后开始对话"""
        logger.info(reply)
        return reply
    except Exception as e:
        logger.error(e)
        reply = """模型未成功加载"""
        if str(e) == "Unknown platform: darwin":
            logger.info(
                "该报错可能因为您使用的是 macOS 操作系统，需先下载模型至本地后执行 Web UI，具体方法请参考项目 README 中本地部署方法及常见问题："
                " https://github.com/imClumsyPanda/langchain-ChatGLM"
            )
        else:
            logger.info(reply)
        return reply


def change_mode(mode, history):
    if mode == "知识库问答":
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
# 🎉 BAAI-RAG !!! 🎉
👍 [实现代码点击链接](https://github.com/FlagAI-Open/Aquila2/tree/main/examples)
"""

init_message = f""" 欢迎使用 BAAI-RAG系统!!!
"""

# 初始化消息
model_status = init_model()

default_theme_args = dict(
    font=["Source Sans Pro", "ui-sans-serif", "system-ui", "sans-serif"],
    font_mono=["IBM Plex Mono", "ui-monospace", "Consolas", "monospace"],
)


with gr.Blocks(css=block_css, theme=gr.themes.Default(**default_theme_args)) as demo:
    file_status, model_status = gr.State(""), gr.State(model_status)
    gr.Markdown(webui_title)
    with gr.Tab("知识库问答"):
        with gr.Row():
            with gr.Column(scale=15):
                chatbot = gr.Chatbot(
                    [[None, init_message], [None, model_status.value]],
                    elem_id="chat-box",
                    show_label=False,
                ).style(height=600)
                query = gr.Textbox(
                    show_label=False, placeholder="请输入提问内容，按回车进行提交"
                ).style(container=False)
                # 这里获取query
                clear_btn = gr.Button(
                    value="新对话", interactive=True, elem_id="clear_btn"
                )

            with gr.Column(scale=5):
                mode = gr.Radio(
                    ["知识库问答"],
                    label="请选择使用模式",
                    value="知识库问答",
                )

                examples = [
                    "Extracting Automata from Recurrent Neural Networks Using Queries and Counterexamples",
                    "Consistent Individualized Feature Attribution for Tree Ensembles",
                ]

                gr.Examples(examples, query, examples_per_page=20)

                flag_csv_logger.setup([query, chatbot, mode], "flagged")
                mode.change(fn=change_mode, inputs=[mode, chatbot], outputs=[chatbot])
                query.submit(get_answer, [query, chatbot, mode], [chatbot, query])
                clear_btn.click(clear_history, None, [chatbot, query])


(
    demo.queue(concurrency_count=3).launch(
        server_name="0.0.0.0",
        server_port=9988,
        show_api=False,
        share=False,
        inbrowser=False,
    )
)
