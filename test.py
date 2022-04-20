import json
import sys

import torch
from torch.nn import functional
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from algo import Model
from utils.dataset import MyDataset
import argparse
import codecs
from sklearn import metrics

sys.stdin = codecs.open("sit.csv", 'r', encoding='utf-8')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_model_file', type=str, default='out_model_file')
    args = parser.parse_args()
    model = Model().cuda()
    model.load_state_dict(torch.load(args.in_model_file))
    queue = []
    window_size = 10
    label2id, id2label = MyDataset.convert2id()
    i = 0
    predict, gold = [], []
    while(True):
        x = sys.stdin.readline()
        if len(x) == 0:
            break
        x = [float(i) for i in x.split(',')]
        assert len(x) == 90
        queue.append(x)
        if len(queue) < window_size:
            continue
        sample = torch.tensor(queue).cuda()
        with torch.no_grad():
            pred_id = model(sample.unsqueeze(0)).argmax().item()
            gold.append(label2id['downstair'])
            predict.append(pred_id)
            pred = id2label[pred_id]
        print(i)
        i += 1

    print(metrics.classification_report(gold, predict))