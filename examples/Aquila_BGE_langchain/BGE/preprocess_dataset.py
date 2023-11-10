import json

import argparse

def main(args):
    data_list = []
    with open(args.input_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                data_list.append(data)
    new_data_list = []
    length = 0
    for d in data_list:
        if d['abstract'] is not None and d['abstract'] != '':
            new_data_list.append(d)
            length += 1
    with open(args.output_path, 'w') as f:
        json.dump(new_data_list, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str, default="ai.json")
    parser.add_argument("--output-path", type=str, default="ai_filter.json")
    args = parser.parse_args()
    main(args)