import logging
import time
from sklearn.ensemble import VotingClassifier
import joblib
from sklearn import metrics
from torch.utils.data import DataLoader
import numpy as np
import utils
from utils import CustomDataset
import copy

logger = logging.getLogger("north_star")


class Trainer:
    def __init__(self, model, epochs: int, data_file: str, out_model_file: str, in_model_file: str = None, debug: bool = False):
        self.epochs: int = epochs
        self.data_file = data_file
        self.out_model_file: str = out_model_file
        self.model_num = 10
        self.model_list = [copy.copy(model) for _ in range(self.model_num)]
        if in_model_file is not None:
            self.model_list = joblib.load(in_model_file)
        self.debug = debug

    def train(self):
        feature_list = self.get_train_dataloader()
        train_start = time.perf_counter()
        for i, (X, Y) in enumerate(feature_list):
            self.model_list[i].fit(X, Y)
        train_end = time.perf_counter()
        if self.debug:
            print(train_end - train_start)
        joblib.dump(self.model_list, self.out_model_file)

        if self.debug:
            X, golds = self.get_eval_dataloader()
            test_start = time.perf_counter()
            predicts = list(map(lambda x: x.predict(X), self.model_list))
            predicts = np.array(predicts).transpose()
            predicts = [np.argmax(np.bincount(i)) for i in predicts]
            test_end = time.perf_counter()
            print(test_end - test_start)
            logger.info("\n" + metrics.classification_report(golds, predicts))
            logger.info("\n" + str(metrics.confusion_matrix(golds, predicts)))

    def get_train_dataloader(self):
        dataset = CustomDataset(self.data_file, window_size=utils.window_size)
        dataloader = DataLoader(dataset, batch_size=len(dataset)//self.model_num, shuffle=True)
        return_features = []
        for i, (X, Y) in enumerate(dataloader):
            if len(Y) < len(dataset)//self.model_num:
                break
            X = X.reshape(X.shape[0], -1).numpy()
            return_features.append((X, Y.numpy()))
        return return_features

    def get_eval_dataloader(self):
        dataset = CustomDataset('../eval', window_size=utils.window_size)
        dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=True)
        X, Y = next(iter(dataloader))
        X = X.reshape(len(dataset), -1).numpy()
        return X, Y

    def __call__(self):
        return self.train()
