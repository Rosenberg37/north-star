import argparse
import codecs
import sys

import torch
from sklearn import metrics

from algo import Model
from utils.dataset import CustomDataset

sys.stdin = codecs.open("whatever.csv", 'r', encoding='utf-8')


class Predictor:
    def __init__(self, window_size: int):
        self.in_model_file = None
        self.window_size = window_size
        self.model = self.model_init()

    def predict(self):
        model = Model().cuda()
        model.load_state_dict(torch.load(self.in_model_file))

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
                continue
            sample = torch.tensor(queue).cuda()
            with torch.no_grad():
                predict_id = model(sample.unsqueeze(0)).argmax().item()
                gold.append(CustomDataset.label2id['downstairs'])
                predict.append(predict_id)

        print(metrics.classification_report(gold, predict))

    def parse(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--in_model_file', type=str, default='out_model_file')
        self.__dict__.update(parser.parse_args().__dict__)
        return self

    def model_init(self):
        self.model = Model().cuda()
        self.model.load_state_dict(torch.load(self.in_model_file))
        return self.model

    def __call__(self):
        return self.predict()


if __name__ == "__main__":
    predictor = Predictor(10)
    predictor.parse()
    predictor()
