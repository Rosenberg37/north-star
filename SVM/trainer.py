from sklearn import neighbors
import joblib
from tqdm import tqdm
import utils
from utils import CustomDataset
from torch.utils.data import DataLoader
from sklearn import metrics
import logging

logger = logging.getLogger("north_star")
class Trainer:
    def __init__(self, epochs: int, data_file: str, out_model_file: str, in_model_file: str = None, debug: bool = False):
        self.epochs: int = epochs
        self.data_file = data_file
        self.out_model_file: str = out_model_file
        self.model = neighbors.KNeighborsClassifier()
        if in_model_file is not None:
            self.model = joblib.load(in_model_file)
        self.debug = debug

    def train(self):
        X, Y = self.get_train_dataloader()
        self.model.fit(X, Y)
        #joblib.dump(self.model, self.out_model_file)

        if self.debug:
            X, golds = self.get_eval_dataloader()
            predicts = self.model.predict(X)
            logger.info("\n" + metrics.classification_report(golds, predicts))
            logger.info("\n" + str(metrics.confusion_matrix(golds, predicts)))

    def get_train_dataloader(self):
        dataset = CustomDataset(self.data_file, window_size=utils.window_size)
        dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=True)
        X, Y = next(iter(dataloader))
        X = X.reshape(len(dataset), -1).numpy()
        return X, Y

    def get_eval_dataloader(self):
        dataset = CustomDataset('eval', window_size=utils.window_size)
        dataloader = DataLoader(dataset, batch_size=1000, shuffle=True)
        X, Y = next(iter(dataloader))
        X = X.reshape(1000, -1).numpy()
        return X, Y

    def __call__(self):
        return self.train()
