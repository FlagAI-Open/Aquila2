import json
import os

import faiss
import numpy as np
from langchain import PromptTemplate, LLMChain
from langchain.chat_models import ChatOpenAI
from pyserini.index import IndexReader
from pyserini.search import LuceneSearcher
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import torch


os.environ['OPENAI_API_KEY'] = "YOUR-KEY"
class QueryExtractor:
    def __init__(self):
        prompt_template = """Generate an informative query from the following conversation.\nConversation: {text}\nQuery:"""
        llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
        prompt = PromptTemplate(
            input_variables=["text"],
            template=prompt_template,
        )

        self.llm_chain = LLMChain(llm=llm, prompt=prompt)

    def run(self, text):
        question = self.llm_chain.predict(text=text).strip()
        return question


class QuerySummarizer:
    def __init__(self):
        prompt_template = """Summarize the topic of the text.\nQuestion: {question}\nSummary:"""
        llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
        prompt = PromptTemplate(
            input_variables=["question"],
            template=prompt_template,
        )

        self.llm_chain = LLMChain(llm=llm, prompt=prompt)

    def run(self, question):
        question = self.llm_chain.predict(question=question)
        return question


class SearchIndex:
    def __init__(self,
                 data_path: str,
                 abstract_emb_path: str,
                 abstract_index_path: str,
                 abstract_bm25_index_path: str,
                 meta_emb_path: str,
                 meta_index_path: str,
                 meta_bm25_index_path: str,
                 batch_size: int=250):
        self.abstract_doc_emb = np.load(abstract_emb_path)
        if not os.path.exists(abstract_index_path):
            self.abstract_index = faiss.IndexFlatIP(self.abstract_doc_emb.shape[1])
            self.abstract_index.add(self.abstract_doc_emb)
            faiss.write_index(self.abstract_index, abstract_index_path)
        else:
            self.abstract_index = faiss.read_index(abstract_index_path)
        self.abstract_bm25_searcher = LuceneSearcher(abstract_bm25_index_path)
        self.abstract_index_reader = IndexReader(abstract_bm25_index_path)

        self.meta_doc_emb = np.load(meta_emb_path)
        if not os.path.exists(meta_index_path):
            self.meta_index = faiss.IndexFlatIP(self.meta_doc_emb.shape[1])
            self.meta_index.add(self.meta_doc_emb)
            faiss.write_index(self.meta_index, meta_index_path)
        else:
            self.meta_index = faiss.read_index(meta_index_path)
        self.meta_bm25_searcher = LuceneSearcher(meta_bm25_index_path)
        self.meta_index_reader = IndexReader(meta_bm25_index_path)

        self.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained('../BGE/bge-large-en-v1.5')
        self.model = AutoModel.from_pretrained('../BGE/bge-large-en-v1.5')
        self.model = self.model.to(self.device)
        self.rerank_tokenizer = AutoTokenizer.from_pretrained('../BGE/bge-reranker-large')
        self.rerank_model = AutoModelForSequenceClassification.from_pretrained('../BGE/bge-reranker-large')
        self.rerank_model = self.rerank_model.to(self.device)
        self.rerank_model.eval()
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        self.batch_size = batch_size

    def search_original(self, query: str, retrieval_type: str = 'semantic', num: int = 5, rerank: str = "disable", rerank_num: int = 5):
        if rerank == 'enable':
            if retrieval_type != 'merge':
                if retrieval_type == 'semantic':
                    query = f'Represent this sentence for searching relevant passages: {query}'
                    encoded_input = self.tokenizer(query, max_length=512, padding=True, truncation=True, return_tensors='pt').to(self.device)

                    with torch.no_grad():
                        model_output = self.model(**encoded_input)
                        query_embedding = model_output[0][:, 0]
                    query_embedding = torch.nn.functional.normalize(query_embedding, p=2, dim=1).detach().cpu().numpy()
                    D, I = self.abstract_index.search(query_embedding, rerank_num)
                    distances, ids = D[0], I[0]
                else:
                    hits = self.abstract_bm25_searcher.search(query, rerank_num)
                    distances, ids = [int(e.score) for e in hits], [int(e.docid) for e in hits]
                pairs = []
                for i in ids:
                    pairs.append([query, self.data[i]['abstract']])
                with torch.no_grad():
                    all_scores = []
                    for i in range(0, len(pairs), self.batch_size):
                        batch_pairs = pairs[i:i + self.batch_size]

                        inputs = self.rerank_tokenizer(batch_pairs, padding=True, truncation=True, return_tensors='pt',
                                                       max_length=512).to(self.device)
                        scores = self.rerank_model(**inputs, return_dict=True).logits.view(
                            -1, ).float().detach().cpu().numpy()
                        all_scores.extend(scores)
                    sorted_indices = np.argsort(all_scores)[::-1]
                result_dict = {}
                for i in range(num):
                    if retrieval_type == 'semantic':
                        result_dict[str(len(result_dict.keys()))] = {
                            'original semantic score': str(distances[sorted_indices[i]]),
                            'rerank score': str(all_scores[sorted_indices[i]]),
                            'content': self.data[ids[sorted_indices[i]]]}
                    else:
                        result_dict[str(len(result_dict.keys()))] = {
                            'original term frequency score': str(distances[sorted_indices[i]]),
                            'rerank score': str(all_scores[sorted_indices[i]]),
                            'content': self.data[ids[sorted_indices[i]]]}
            else:
                query = f'Represent this sentence for searching relevant passages: {query}'
                encoded_input = self.tokenizer(query, max_length=512, padding=True, truncation=True,
                                               return_tensors='pt').to(self.device)

                with torch.no_grad():
                    model_output = self.model(**encoded_input)
                    query_embedding = model_output[0][:, 0]
                query_embedding = torch.nn.functional.normalize(query_embedding, p=2, dim=1).detach().cpu().numpy()
                D, I = self.abstract_index.search(query_embedding, rerank_num)
                distances, ids = list(D[0]), list(I[0])
                hits = self.abstract_bm25_searcher.search(query, rerank_num)
                distances.extend([int(e.score) for e in hits])
                ids.extend([int(e.docid) for e in hits])
                ids = list(set(ids))
                # distances = [distances[i] for i in [new_ids.index(e) for e in ids]]
                # ids = new_ids
                pairs = []
                for i in ids:
                    pairs.append([query, self.data[i]['abstract']])
                with torch.no_grad():
                    all_scores = []
                    for i in range(0, len(pairs), self.batch_size):
                        batch_pairs = pairs[i:i + self.batch_size]

                        inputs = self.rerank_tokenizer(batch_pairs, padding=True, truncation=True, return_tensors='pt',
                                                       max_length=512).to(self.device)
                        scores = self.rerank_model(**inputs, return_dict=True).logits.view(
                            -1, ).float().detach().cpu().numpy()
                        all_scores.extend(scores)
                    sorted_indices = np.argsort(all_scores)[::-1]
                result_dict = {}
                for i in range(num):
                    result_dict[str(len(result_dict.keys()))] = {'rerank score': str(all_scores[sorted_indices[i]]),
                                                                 'content': self.data[ids[sorted_indices[i]]]}
        else:
            if retrieval_type != 'merge':
                if retrieval_type == 'semantic':
                    query = f'Represent this sentence for searching relevant passages: {query}'
                    encoded_input = self.tokenizer(query, max_length=512, padding=True, truncation=True, return_tensors='pt').to(self.device)

                    with torch.no_grad():
                        model_output = self.model(**encoded_input)
                        query_embedding = model_output[0][:, 0]
                    query_embedding = torch.nn.functional.normalize(query_embedding, p=2, dim=1).detach().cpu().numpy()
                    D, I = self.abstract_index.search(query_embedding, num)
                    distances, ids = D[0], I[0]
                else:
                    hits = self.abstract_bm25_searcher.search(query, num)
                    distances, ids = [int(e.score) for e in hits], [int(e.docid) for e in hits]
                result_dict = {}
                for i in range(num):
                    if retrieval_type == 'semantic':
                        result_dict[str(len(result_dict.keys()))] = {'original semantic score': str(distances[i]),
                                                                     'content': self.data[ids[i]]}
                    else:
                        result_dict[str(len(result_dict.keys()))] = {'original term frequency score': str(distances[i]),
                                                                     'content': self.data[ids[i]]}
            else:
                query = f'Represent this sentence for searching relevant passages: {query}'
                encoded_input = self.tokenizer(query, max_length=512, padding=True, truncation=True,
                                               return_tensors='pt').to(self.device)

                with torch.no_grad():
                    model_output = self.model(**encoded_input)
                    query_embedding = model_output[0][:, 0]
                query_embedding = torch.nn.functional.normalize(query_embedding, p=2, dim=1).detach().cpu().numpy()
                D, I = self.abstract_index.search(query_embedding, num)
                original_dict = {}
                for d, i in zip(D[0], I[0]):
                    original_dict[int(i)] = {'original semantic score': d}
                hits = self.abstract_bm25_searcher.search(query, num)
                for d, i in zip([int(e.score) for e in hits], [int(e.docid) for e in hits]):
                    if original_dict.get(i) is not None:
                        original_dict[i]['original term frequency score'] = d
                    else:
                        original_dict[i] = {'original term frequency score': d}
                for k in original_dict.keys():
                    if original_dict[k].get('original semantic score') is None:
                        original_dict[k]['original semantic score'] = (query_embedding @ self.abstract_doc_emb[k].T)[0]
                    if original_dict[k].get('original term frequency score') is None:
                        original_dict[k]['original term frequency score'] = self.abstract_index_reader.compute_bm25_term_weight(str(k), query, analyzer=None)
                    original_dict[k]['merge score'] = float(original_dict[k]['original term frequency score']) * 0.2 + float(original_dict[k]['original semantic score'])
                sorted_data = dict(sorted(original_dict.items(), key=lambda x: x[1]["merge score"], reverse=True))
                result_dict = {}
                all_keys = list(sorted_data.keys())
                for i in range(num):
                    result_dict[str(len(result_dict.keys()))] = {'original semantic score': str(sorted_data[all_keys[i]]['original semantic score']),
                                                                 'original term frequency score': str(sorted_data[all_keys[i]]['original term frequency score']),
                                                                 'merge score': sorted_data[all_keys[i]]['merge score'],
                                                                 'content': self.data[all_keys[i]]}
        torch.cuda.empty_cache()
        return result_dict

    def search_conditional(self, query: str, retrieval_type: str = 'semantic', num: int = 5, rerank: str = "disable", rerank_num: int = 5):
        if rerank == 'enable':
            if retrieval_type != 'merge':
                if retrieval_type == 'semantic':
                    query = f'Represent this sentence for searching relevant passages: {query}'
                    encoded_input = self.tokenizer(query, max_length=512, padding=True, truncation=True,
                                                   return_tensors='pt').to(self.device)

                    with torch.no_grad():
                        model_output = self.model(**encoded_input)
                        query_embedding = model_output[0][:, 0]
                    query_embedding = torch.nn.functional.normalize(query_embedding, p=2,
                                                                    dim=1).detach().cpu().numpy()
                    D, I = self.meta_index.search(query_embedding, rerank_num)
                    distances, ids = D[0], I[0]
                else:
                    hits = self.meta_bm25_searcher.search(query, rerank_num)
                    distances, ids = [int(e.score) for e in hits], [int(e.docid) for e in hits]
                pairs = []
                for i in ids:
                    pairs.append([query, f"title: {self.data[i]['title']}\nauthors: " + ', '.join(
                        self.data[i]['authors']) + f"\nabstract: {self.data[i]['abstract']}"])
                with torch.no_grad():
                    all_scores = []
                    for i in range(0, len(pairs), self.batch_size):
                        batch_pairs = pairs[i:i + self.batch_size]

                        inputs = self.rerank_tokenizer(batch_pairs, padding=True, truncation=True,
                                                       return_tensors='pt',
                                                       max_length=512).to(self.device)
                        scores = self.rerank_model(**inputs, return_dict=True).logits.view(
                            -1, ).float().detach().cpu().numpy()
                        all_scores.extend(scores)
                    sorted_indices = np.argsort(all_scores)[::-1]
                result_dict = {}
                for i in range(num):
                    if retrieval_type == 'semantic':
                        result_dict[str(len(result_dict.keys()))] = {
                            'original semantic score': str(distances[sorted_indices[i]]),
                            'rerank score': str(all_scores[sorted_indices[i]]),
                            'content': self.data[ids[sorted_indices[i]]]}
                    else:
                        result_dict[str(len(result_dict.keys()))] = {
                            'original term frequency score': str(distances[sorted_indices[i]]),
                            'rerank score': str(all_scores[sorted_indices[i]]),
                            'content': self.data[ids[sorted_indices[i]]]}
            else:
                query = f'Represent this sentence for searching relevant passages: {query}'
                encoded_input = self.tokenizer(query, max_length=512, padding=True, truncation=True,
                                               return_tensors='pt').to(self.device)

                with torch.no_grad():
                    model_output = self.model(**encoded_input)
                    query_embedding = model_output[0][:, 0]
                query_embedding = torch.nn.functional.normalize(query_embedding, p=2,
                                                                dim=1).detach().cpu().numpy()
                D, I = self.meta_index.search(query_embedding, rerank_num)
                distances, ids = list(D[0]), list(I[0])
                hits = self.meta_bm25_searcher.search(query, rerank_num)
                distances.extend([int(e.score) for e in hits])
                ids.extend([int(e.docid) for e in hits])
                ids = list(set(ids))
                # distances = [distances[i] for i in [new_ids.index(e) for e in ids]]
                # ids = new_ids
                pairs = []
                for i in ids:
                    pairs.append([query, f"title: {self.data[i]['title']}\nauthors: " + ', '.join(
                        self.data[i]['authors']) + f"\nabstract: {self.data[i]['abstract']}"])
                with torch.no_grad():
                    all_scores = []
                    for i in range(0, len(pairs), self.batch_size):
                        batch_pairs = pairs[i:i + self.batch_size]

                        inputs = self.rerank_tokenizer(batch_pairs, padding=True, truncation=True,
                                                       return_tensors='pt',
                                                       max_length=512).to(self.device)
                        scores = self.rerank_model(**inputs, return_dict=True).logits.view(
                            -1, ).float().detach().cpu().numpy()
                        all_scores.extend(scores)
                    sorted_indices = np.argsort(all_scores)[::-1]
                result_dict = {}
                for i in range(num):
                    result_dict[str(len(result_dict.keys()))] = {
                        'rerank score': str(all_scores[sorted_indices[i]]),
                        'content': self.data[ids[sorted_indices[i]]]}
        else:
            if retrieval_type != 'merge':
                if retrieval_type == 'semantic':
                    query = f'Represent this sentence for searching relevant passages: {query}'
                    encoded_input = self.tokenizer(query, max_length=512, padding=True, truncation=True,
                                                   return_tensors='pt').to(self.device)

                    with torch.no_grad():
                        model_output = self.model(**encoded_input)
                        query_embedding = model_output[0][:, 0]
                    query_embedding = torch.nn.functional.normalize(query_embedding, p=2,
                                                                    dim=1).detach().cpu().numpy()
                    D, I = self.meta_index.search(query_embedding, num)
                    distances, ids = D[0], I[0]
                else:
                    hits = self.meta_bm25_searcher.search(query, num)
                    distances, ids = [int(e.score) for e in hits], [int(e.docid) for e in hits]
                result_dict = {}
                for i in range(num):
                    if retrieval_type == 'semantic':
                        result_dict[str(len(result_dict.keys()))] = {
                            'original semantic score': str(distances[i]),
                            'content': self.data[ids[i]]}
                    else:
                        result_dict[str(len(result_dict.keys()))] = {
                            'original term frequency score': str(distances[i]),
                            'content': self.data[ids[i]]}
            else:
                query = f'Represent this sentence for searching relevant passages: {query}'
                encoded_input = self.tokenizer(query, max_length=512, padding=True, truncation=True,
                                               return_tensors='pt').to(self.device)

                with torch.no_grad():
                    model_output = self.model(**encoded_input)
                    query_embedding = model_output[0][:, 0]
                query_embedding = torch.nn.functional.normalize(query_embedding, p=2,
                                                                dim=1).detach().cpu().numpy()
                D, I = self.meta_index.search(query_embedding, num)
                original_dict = {}
                for d, i in zip(D[0], I[0]):
                    original_dict[int(i)] = {'original semantic score': d}
                hits = self.meta_bm25_searcher.search(query, num)
                for d, i in zip([int(e.score) for e in hits], [int(e.docid) for e in hits]):
                    if original_dict.get(i) is not None:
                        original_dict[i]['original term frequency score'] = d
                    else:
                        original_dict[i] = {'original term frequency score': d}
                for k in original_dict.keys():
                    if original_dict[k].get('original semantic score') is None:
                        original_dict[k]['original semantic score'] = (query_embedding @ self.meta_doc_emb[k].T)[0]
                    if original_dict[k].get('original term frequency score') is None:
                        original_dict[k]['original term frequency score'] = self.meta_index_reader.compute_bm25_term_weight(str(k), query, analyzer=None)
                    original_dict[k]['merge score'] = float(
                            original_dict[k]['original term frequency score']) * 0.2 + float(
                            original_dict[k]['original semantic score'])
                sorted_data = dict(
                    sorted(original_dict.items(), key=lambda x: x[1]["merge score"], reverse=True))
                result_dict = {}
                all_keys = list(sorted_data.keys())
                for i in range(num):
                    result_dict[str(len(result_dict.keys()))] = {
                        'original semantic score': str(sorted_data[all_keys[i]]['original semantic score']),
                        'original term frequency score': str(
                            sorted_data[all_keys[i]]['original term frequency score']),
                        'merge score': sorted_data[all_keys[i]]['merge score'],
                        'content': self.data[all_keys[i]]}
        torch.cuda.empty_cache()
        return result_dict


class SearchTool:
    def __init__(self,
                 data_path: str,
                 abstract_emb_path: str,
                 abstract_index_path: str,
                 abstract_bm25_index_path: str,
                 meta_emb_path: str,
                 meta_index_path: str,
                 meta_bm25_index_path: str,
                 batch_size: int = 128):
        self.search_index = SearchIndex(data_path,
                                        abstract_emb_path,
                                        abstract_index_path,
                                        abstract_bm25_index_path,
                                        meta_emb_path,
                                        meta_index_path,
                                        meta_bm25_index_path,
                                        batch_size)
        self.query_extractor = QueryExtractor()
        self.query_summarizer = QuerySummarizer()
        #self.log = open('log.txt', 'a')

    def conversation_search(self, text: str, retrieval_type: str = 'semantic', target: str = 'original', num: int = 5, rerank: str = 'disable', rerank_num: int = 5):
        query = self.query_extractor.run(text)
        #self.log.write(f'Text: {text}\nQuery: {query}\nQuery Type: conversation\nTarget: {target}\n\n')
        #self.log.flush()
        if target == 'original':
            return self.search_index.search_original(query, retrieval_type, num, rerank, rerank_num)
        else:
            return self.search_index.search_conditional(query, retrieval_type, num, rerank, rerank_num)

    def topic_search(self, text: str, retrieval_type: str = 'semantic', target: str = 'original', num: int = 5, rerank: str = 'disable', rerank_num: int = 5):
        query = self.query_summarizer.run(text)
        #self.log.write(f'Text: {text}\nQuery: {query}\nQuery Type: topic\nTarget: {target}\n\n')
        #self.log.flush()
        if target == 'original':
            return self.search_index.search_original(query, retrieval_type, num, rerank, rerank_num)
        else:
            return self.search_index.search_conditional(query, retrieval_type, num, rerank, rerank_num)

    def query_search(self, text: str, retrieval_type: str = 'semantic', target: str = 'original', num: int = 5, rerank: str = 'disable', rerank_num: int = 5):
        query = text
        #self.log.write(f'Text: {text}\nQuery: {query}\nQuery Type: query\nTarget: {target}\n\n')
        #self.log.flush()
        if target == 'original':
            return self.search_index.search_original(query, retrieval_type, num, rerank, rerank_num)
        else:
            return self.search_index.search_conditional(query, retrieval_type, num, rerank, rerank_num)

    def search(self, query: str, retrieval_type: str = 'semantic', search_type: str = 'query', target: str = 'original', num: int = 5, rerank: str = 'disable', rerank_num: int = 5):
        if search_type == 'by conversation':
            return self.conversation_search(query, retrieval_type, target, num, rerank, rerank_num)
        elif search_type == 'by topic':
            return self.topic_search(query, retrieval_type, target, num, rerank, rerank_num)
        else:
            return self.query_search(query, retrieval_type, target, num, rerank, rerank_num)
