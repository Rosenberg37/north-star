import argparse
import codecs
import sys

import torch
from sklearn import metrics

from algo import Model
from utils.dataset import CustomDataset

sys.stdin = codecs.open("whatever.csv", 'r', encoding='utf-8')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_model_file', type=str, default='out_model_file')
    args = parser.parse_args()

    model = Model().cuda()
    model.load_state_dict(torch.load(args.in_model_file))

    queue = []
    window_size = 10
    label2id, id2label = CustomDataset.convert2id()
    predict, gold = [], []
    while True:
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
            predict_id = model(sample.unsqueeze(0)).argmax().item()
            gold.append(label2id['downstairs'])
            predict.append(predict_id)
            predict_label = id2label[predict_id]

    print(metrics.classification_report(gold, predict))
