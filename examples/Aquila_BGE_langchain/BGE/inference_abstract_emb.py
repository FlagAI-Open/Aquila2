import json
import pickle

import argparse
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset, DistributedSampler, SequentialSampler
import torch.distributed as dist
from torch_geometric.data import DataLoader
from tqdm import tqdm
from transformers import PreTrainedTokenizer, AutoModel

import numpy as np

from transformers import AutoTokenizer


class EmbDataset(Dataset):
    def __init__(
            self,
            tokenizer: PreTrainedTokenizer,
            path: str
    ):
        self.data = []
        self.tokenizer = tokenizer

        with open(path, 'r') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        try:
            texts = self.data[item]['abstract']
            batch_dict = self.tokenizer(texts, max_length=512, padding=True, truncation=True, return_tensors='pt')
        except:
            print(self.data[item])
        attention_mask = batch_dict['attention_mask'][0].tolist() + [0] * (512 - len(batch_dict['attention_mask'][0]))
        token_type_ids = batch_dict['token_type_ids'][0].tolist() + [0] * (512 - len(batch_dict['token_type_ids'][0]))
        input_ids = batch_dict['input_ids'][0].tolist() + [0] * (512 - len(batch_dict['token_type_ids'][0]))

        return torch.LongTensor(input_ids), torch.LongTensor(token_type_ids), torch.LongTensor(attention_mask)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    tokenizer = AutoTokenizer.from_pretrained('../BGE/bge-large-en-v1.5')

    model = AutoModel.from_pretrained('../BGE/bge-large-en-v1.5').to(device)
    model = torch.nn.parallel.DataParallel(model)

    dataset = EmbDataset(tokenizer, args.input_path)
    loader = DataLoader(dataset=dataset, batch_size=args.batch_size, sampler=SequentialSampler(dataset), shuffle=False,
                        drop_last=False, num_workers=6)

    model.eval()
    existing_data = []
    for step, data in enumerate(tqdm(loader, total=len(loader))):
        input_ids, token_type_ids, attention_mask = data
        input_ids = input_ids.to(device)
        token_type_ids = token_type_ids.to(device)
        attention_mask = attention_mask.to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)[0][:, 0]
            batch_vecs = torch.nn.functional.normalize(outputs, p=2, dim=1).detach().cpu().numpy()

            existing_data.append(batch_vecs)

    np.save(args.output_path, np.concatenate(existing_data))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str, default="ai_filter.json")
    parser.add_argument("--output-path", type=str, default="abstract.npy")
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()
    main(args)
