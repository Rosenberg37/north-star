import codecs

import utils
from sklearn import neighbors
import joblib
from tqdm import tqdm
import utils
from utils import CustomDataset
from torch.utils.data import DataLoader
from sklearn import metrics
import logging
import sys
import numpy as np

sys.stdin = codecs.open('dlf_stand.csv')

class Predictor:
    def __init__(self, in_model_file: str):
        self.model = neighbors.KNeighborsClassifier()
        if in_model_file is not None:
            self.model = joblib.load(in_model_file)

    def predict(self):
        queue = []
        while True:
            x = sys.stdin.readline()
            if len(x) == 0:
                break
            try:
                x = [float(i) for i in x.split(',')]
            except ValueError as _:
                raise ValueError(f"Please check the input format, making sure it is separated by , in Ascii.")
            if len(x) != utils.data_size:
                raise RuntimeError(f"Unexpected data input length. Expected:{utils.data_size}")
            queue.append(x)

            if len(queue) < utils.window_size:
                continue

            sample = np.array(queue)
            sample = np.reshape(sample, (1, -1))

            print(self.model.predict(sample))
            queue.pop(0)

    def __call__(self):
        return self.predict()