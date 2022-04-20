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


sys.stdin = codecs.open("sit.csv", 'r', encoding='utf-8')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_model_file', type=str, default='out_model_file')
    args = parser.parse_args()
    model = Model().cuda()
    model.load_state_dict(torch.load(args.in_model_file))
    queue = []
    window_size = 10
    _, id2label = MyDataset.convert2id()
    while(True):
        x = sys.stdin.readline()
        x = [float(i) for i in x.split(',')]
        assert len(x) == 90
        queue.append(x)
        if len(queue) < window_size:
            continue
        sample = torch.tensor(queue).cuda()
        with torch.no_grad():
            pred_tensor = model(sample.unsqueeze(0))
            pred = id2label[pred_tensor.argmax().item()]
            print(pred)


