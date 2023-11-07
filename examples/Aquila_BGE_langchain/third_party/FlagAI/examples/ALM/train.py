# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
import pandas as pd
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from flagai.auto_model.auto_loader import AutoLoader
from flagai.trainer import Trainer
from tqdm import tqdm 
from rouge_score import rouge_scorer
from torch import argmax
import sacrebleu


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
trainer = Trainer(
    env_type="pytorch",
    experiment_name="ALM_seq2seq",
    batch_size=1,
    gradient_accumulation_steps=1,
    lr=1e-5,
    weight_decay=1e-5,
    epochs=1,
    log_interval=10,
    eval_interval=10,
    load_dir=None,
    pytorch_device=device,
    save_dir="checkpoints_alm_title_generation",
    save_interval=1000,
    num_checkpoints=1,
)

data_dir = './data/train.tsv'
val_dir = './data/valid.tsv'



maxlen = 256
auto_loader = AutoLoader("lm",
                         model_name="ALM-1.0",)
model = auto_loader.get_model()
tokenizer = auto_loader.get_tokenizer()


def read_file(path):
    src = []
    tgt = []
    df = pd.read_csv(path, sep="\t")
    for idx, row in tqdm(df.iterrows()):
        src.append(row["source"])
        tgt.append(row["target"])
    return src, tgt


def bleu_metric(predictions, labels, meta=None, metric="rouge-1", duplicate_rate=0.7, dataset='cnn_dm'):
    ref_list = []
    for i in labels:
        i = i.tolist()
        ref = tokenizer.DecodeIds(i)
        ref_list.append(ref)
    pred_list = []

    for prediction in predictions:
        buf = []
        prediction = prediction.tolist()
        prediction = tokenizer.DecodeIds(prediction)
        pred_list.append(prediction)

    bleu_results = sacrebleu.corpus_bleu(pred_list, [ref_list])
    bleu_score = bleu_results.score
    return bleu_score

def rouge_metric(predictions, labels, meta=None, metric="rouge-1", duplicate_rate=0.7, dataset='cnn_dm'):
    metric_dict = {"rouge-1": "rouge1", "rouge-2": "rouge2", "rouge-l": "rougeLsum"}
    ref_list = []
    for i in labels:
        i = i.tolist()
        ref = tokenizer.DecodeIds(i)
        ref_list.append(ref)
    pred_list = []
    for prediction in predictions:
        buf = []
        prediction = prediction.tolist()
        prediction = tokenizer.DecodeIds(prediction)
        pred_list.append(prediction)
    scorer = rouge_scorer.RougeScorer([metric_dict[metric]], use_stemmer=True)
    scores = [scorer.score(pred, ref) for pred, ref in zip(pred_list, ref_list)]
    scores = [score[metric_dict[metric]].fmeasure * 100 for score in scores]
    scores = sum(scores) / len(scores)
    return scores



class ALMSeq2seqDataset(Dataset):

    def __init__(self,
                 sents_src,
                 sents_tgt,
                 tokenizer,
                 max_src_length=512,
                 max_tgt_length=512):
        super(ALMSeq2seqDataset, self).__init__()
        self.sents_src = sents_src
        self.sents_tgt = sents_tgt
        self.tokenizer = tokenizer
        self.max_src_length = max_src_length
        self.max_tgt_length = max_tgt_length
        self.no_block_position = False

    def __getitem__(self, i):
        source_text = self.sents_src[i]
        target_text = self.sents_tgt[i]
        data = self.tokenizer.encode_plus(source_text=source_text, target_text=target_text, max_length=512)

        return data

    def __len__(self):

        return len(self.sents_src)


class ALMCollateFN():  #padding process in each batch

    def __init__(self, pad_id):
        self.pad_id = pad_id

    def pad_token(self, tokens, max_length):
        pad_len = max_length - len(tokens)
        tokens += [self.pad_id] * pad_len
        return tokens

    def pad_position_ids(self, position_ids, max_length):
        pad_len = max_length - len(position_ids[0])
        position_ids[0] += [len(position_ids[0]) + x for x in range(pad_len)]
        position_ids[1] += [1] * pad_len
        return position_ids

    def pad_loss_mask(self, loss_mask, max_length):
        pad_len = max_length - len(loss_mask)
        loss_mask += [0] * pad_len
        return loss_mask

    def __call__(self, batch):
        input_ids = [data["input_ids"] for data in batch]
        target_ids = [data["target_ids"] for data in batch]
        position_ids = [data["position_ids"] for data in batch]
        attention_mask = [data['attention_mask'] for data in batch]
        loss_mask = [data['loss_mask'] for data in batch]

        max_length = max([len(t) for t in input_ids])
        for i in range(len(input_ids)):
            input_ids[i] = self.pad_token(input_ids[i], max_length)
            target_ids[i] = self.pad_token(target_ids[i], max_length)
            position_ids[i] = self.pad_position_ids(position_ids[i],
                                                    max_length)
            loss_mask[i] = self.pad_loss_mask(loss_mask[i], max_length)
        return {
            'input_ids': torch.LongTensor(input_ids),
            'labels': torch.LongTensor(target_ids),
            'position_ids': torch.LongTensor(position_ids),
            'attention_mask': torch.LongTensor(attention_mask),
            'loss_mask': torch.LongTensor(loss_mask)
        }


sents_src, sents_tgt = read_file(data_dir)
vail_src, vail_tgt = read_file(val_dir)

my_collate_fn = ALMCollateFN(
    pad_id=tokenizer.get_command_id('pad'))

data_len = len(sents_tgt)
train_size = int(data_len * 0.8)
train_src = sents_src
train_tgt = sents_tgt

val_src = vail_src
val_tgt = vail_tgt

train_dataset = ALMSeq2seqDataset(train_src,
                                  train_tgt,
                                  tokenizer=tokenizer)
val_dataset = ALMSeq2seqDataset(val_src,
                                val_tgt,
                                tokenizer=tokenizer)
trainer.train(model,
              train_dataset=train_dataset,
              valid_dataset=val_dataset,
              metric_methods=[['rouge_scorer', rouge_metric], ['bleu', bleu_metric]],
              collate_fn=my_collate_fn)
