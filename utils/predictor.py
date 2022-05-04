import sys

import torch

import utils


class Predictor:
    def __init__(self, in_model_file: str):
        self.model = utils.Model(utils.data_size)
        self.model.load_state_dict(torch.load(in_model_file))

    def predict(self):
        queue = []
        while True:
            x = sys.stdin.readline()
            if len(x) == 0:
                break
            x = [float(i) for i in x.split(',')]
            if len(x) != utils.data_size:
                raise RuntimeError(f"Unexpected data input length. Expected:{utils.data_size}")
            queue.append(x)

            if len(queue) < utils.window_size:
                continue

            sample = torch.tensor(queue)
            with torch.no_grad():
                print(self.model(sample.unsqueeze(0)).argmax().item())
            queue.pop(0)

    def __call__(self):
        return self.predict()
