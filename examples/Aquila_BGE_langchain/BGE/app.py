import argparse
from flask import Flask, request, jsonify
from tool import SearchTool
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

app = Flask(__name__)
search_tool = None
limiter = Limiter(get_remote_address, app=app, default_limits=["50 per minute"])


@app.route('/search', methods=['POST'])
def index():
    global search_tool
    data = request.get_json()
    data = {
    "input": "give me a paper written by Zhiyuan Liu",
    "retrieval_type": "merge",
    "query_type": "by query",
    "target_type": "conditional",
    "num": 10,
    "rerank": "enable",
    "rerank_num": 25
}
    input_text = data.get('input', None)
    retrieval_type = data.get('retrieval_type', 'semantic')
    query_type = data.get('query_type', 'by query')
    target_type = data.get('target_type', 'original')
    num = min(int(data.get('num', 5)), 100)
    num = max(num, 1)
    rerank = data.get('rerank', 'disable')
    rerank_num = min(int(data.get('rerank_num', 100)), 200)
    rerank_num = max(rerank_num, num)
    if input_text is None:
        return jsonify({"error": "No query provided"}), 400
    try:
        print(jsonify(search_tool.search(input_text, retrieval_type, query_type, target_type, num, rerank, rerank_num)))
        return jsonify(search_tool.search(input_text, retrieval_type, query_type, target_type, num, rerank, rerank_num)), 200
    except Exception as e:
        return jsonify({"error": e}), 400




def main(args):
    global search_tool
    search_tool = SearchTool(args.data_path,
                             args.abstract_emb_path,
                             args.abstract_index_path,
                             args.abstract_bm25_index_path,
                             args.meta_emb_path,
                             args.meta_index_path,
                             args.meta_bm25_index_path,
                             args.batch_size)
    global app
    app.run('0.0.0.0', 5000)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="ai_filter.json")
    parser.add_argument("--abstract-emb-path", type=str, default="abstract.npy")
    parser.add_argument("--abstract-index-path", type=str, default="abstract.index")
    parser.add_argument("--abstract-bm25-index-path", type=str, default="abstract_bm25_index")
    parser.add_argument("--meta-emb-path", type=str, default="meta.npy")
    parser.add_argument("--meta-index-path", type=str, default="meta.index")
    parser.add_argument("--meta-bm25-index-path", type=str, default="meta_bm25_index")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    main(args)
