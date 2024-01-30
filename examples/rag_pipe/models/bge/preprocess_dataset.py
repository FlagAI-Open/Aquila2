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

import argparse


## data_format
{
    "arxiv_id": "1805.10616",
    "abstract": "Can evolving networks be inferred and modeled without directly observing\ntheir nodes and edges? In many applications, the edges of a dynamic network\nmight not be observed, but one can observe the dynamics of stochastic cascading\nprocesses (e.g., information diffusion, virus propagation) occurring over the\nunobserved network. While there have been efforts to infer networks based on\nsuch data, providing a generative probabilistic model that is able to identify\nthe underlying time-varying network remains an open question. Here we consider\nthe problem of inferring generative dynamic network models based on network\ncascade diffusion data. We propose a novel framework for providing a\nnon-parametric dynamic network model--based on a mixture of coupled\nhierarchical Dirichlet processes-- based on data capturing cascade node\ninfection times. Our approach allows us to infer the evolving community\nstructure in networks and to obtain an explicit predictive distribution over\nthe edges of the underlying network--including those that were not involved in\ntransmission of any cascade, or are likely to appear in the future. We show the\neffectiveness of our approach using extensive experiments on synthetic as well\nas real-world networks.",
    "title": "Dynamic Network Model from Partial Observations",
    "authors": [
        "Elahe Ghalebi",
        "Baharan Mirzasoleiman",
        "Radu Grosu",
        "Jure Leskovec",
    ],
}


def main(args):
    data_list = []
    with open(args.input_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                data_list.append(data)
    new_data_list = []
    length = 0
    for d in data_list:
        if d.get("abstract", "") or d.get("title", "") or d.get("authors", ""):
            if isinstance(d.get("authors", ""), list):
                d["authors"] = ",".join(d["authors"])
            new_data_list.append(d)
            length += 1
    with open(args.output_path, "w") as f:
        json.dump(new_data_list, f, ensure_ascii=False)
    print(f"process file {args.output_path} done, sample num is {len(new_data_list)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str, default="ai.json")
    parser.add_argument("--output-path", type=str, default="ai_filter.json")
    args = parser.parse_args()
    main(args)
