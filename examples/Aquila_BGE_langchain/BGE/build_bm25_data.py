import json
import os

import argparse


def main(args):
    data = json.load(open(args.data_path))
    abstract = []
    meta = []
    for i,d in enumerate(data):
        abstract.append({'id': i,
                         'contents': d['abstract']})
        meta.append({'id': i,
                     'contents': f"title: {d['title']}\nauthors: " + ', '.join(
                         d['authors']) + f"\nabstract: {d['abstract']}"})
    with open(os.path.join(args.abstract_document_path, 'documents.json'), 'w') as file:
        json.dump(abstract, file)

    with open(os.path.join(args.meta_document_path, 'documents.json'), 'w') as file:
        json.dump(meta, file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="ai_filter.json")
    parser.add_argument("--abstract-document-path", type=str, default="abstract_collection")
    parser.add_argument("--meta-document-path", type=str, default="meta_collection")
    args = parser.parse_args()
    main(args)