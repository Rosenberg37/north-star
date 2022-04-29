import logging

import torch
from sklearn import metrics
from torch.nn import functional
from torch.utils.data import DataLoader
from tqdm import tqdm

import algo
import utils
from utils import CustomDataset

logger = logging.getLogger("north_star")


class Trainer:
    def __init__(self, epochs: int, data_file: str, out_model_file: str, in_model_file: str):
        self.epochs = epochs

        self.out_model_file: str = out_model_file
        self.in_model_file: str = ""

        self.train_dataset = CustomDataset(data_file, window_size=utils.window_size)
        self.eval_dataset = CustomDataset('eval', window_size=utils.window_size)

        self.best = 0
        self.model = algo.Model()
        if in_model_file is not None:
            self.model.load_state_dict(torch.load(in_model_file))
        self.optimizer = self.create_optimizer()

    def train(self):
        dataloader = self.get_train_dataloader()

        for epoch_index in range(self.epochs):
            with tqdm(total=len(dataloader), desc=f"Epoch {epoch_index}") as tbar:
                self.model.train()
                for batch_index, [data, gold] in enumerate(dataloader):
                    self.optimizer.zero_grad()

                    prediction = self.model(data)
                    loss = functional.cross_entropy(prediction, gold)
                    loss.backward()
                    self.optimizer.step()

                    tbar.set_postfix(loss=loss.item())
                    tbar.update()

            with torch.no_grad():
                self.model.eval()
                predicts, golds = [], []
                eval_dataloader = self.get_eval_dataloader()
                for batch_index, [data, gold] in enumerate(eval_dataloader):
                    predicts += self.model(data).argmax(dim=1).tolist()
                    golds += gold.tolist()

                # noinspection PyTypeChecker
                performance = metrics.classification_report(golds, predicts, output_dict=True)['macro avg']['f1-score']
                if performance > self.best:
                    self.best = performance
                    logger.info(f"Save model with best accuracy:{self.best}")
                    torch.save(self.model.state_dict(), self.out_model_file)

                logger.info("\n" + metrics.classification_report(golds, predicts))
                logger.info("\n" + str(metrics.confusion_matrix(golds, predicts)))

    def create_optimizer(self):
        return torch.optim.AdamW(self.model.parameters(), lr=1e-4)

    def get_train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=128, shuffle=True)

    def get_eval_dataloader(self):
        return DataLoader(self.eval_dataset, batch_size=128, shuffle=True)

    def __call__(self):
        return self.train()
