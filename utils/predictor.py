import sys

import torch

import algo
import utils


class Predictor:
    def __init__(self, in_model_file: str):
        self.model = algo.Model()
        self.model.load_state_dict(torch.load(in_model_file))
        self.window_size = utils.window_size
        self.model = None

    def predict(self):
        i = 0
        queue = []
        while True:
            x = sys.stdin.readline()
            if len(x) == 0:
                break
            x = [float(i) for i in x.split(',')]
            assert len(x) == utils.data_size
            queue.append(x)
            if len(queue) < self.window_size:
                i += 1
                continue
            sample = torch.tensor(queue)
            with torch.no_grad():
                print(self.model(sample.unsqueeze(0)).argmax().item())

    def __call__(self):
        return self.predict()
