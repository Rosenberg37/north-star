import sys

import pandas as pd
import torch
from sklearn import metrics

import algo
import utils

gold_file = list(pd.read_csv('ans.csv', header=None).values)


class Predictor:
    def __init__(self, in_model_file: str):
        self.model = algo.Model()
        self.model.load_state_dict(torch.load(in_model_file))
        self.window_size = utils.window_size
        self.model = None

    def predict(self):
        i = 0
        queue = []
        predict, gold = [], []
        while True:
            x = sys.stdin.readline()
            if len(x) == 0:
                break
            x = [float(i) for i in x.split(',')]
            assert len(x) == 90
            queue.append(x)
            if len(queue) < self.window_size:
                i += 1
                continue
            sample = torch.tensor(queue)
            with torch.no_grad():
                predict_id = self.model(sample.unsqueeze(0)).argmax().item()
                gold.append(gold_file[i])
                predict.append(predict_id)
                print("predict: %d   gold: %d" % (predict_id, gold_file[i]))
                i += 1

        print(metrics.classification_report(gold, predict))

    def __call__(self):
        return self.predict()
