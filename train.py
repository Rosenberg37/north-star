import argparse

import torch
from sklearn import metrics
from torch.nn import functional
from torch.utils.data import DataLoader
from tqdm import tqdm

from algo import Model
from utils.dataset import CustomDataset


class Trainer:
    def __init__(self, epochs: int, window_size: int):
        self.window_size = window_size
        self.epoch = epochs
        self.best = 0

        self.data_file = None
        self.out_model_file = None
        self.in_model_file = None

        self.model = self.model_init()
        self.optimizer = self.create_optimizer()

    def parse(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--data_file", type=str, default='data')
        parser.add_argument('--out_model_file', type=str, default='cache/out_model_file.pt')
        parser.add_argument('--in_model_file', type=str, default='None')
        self.__dict__.update(parser.parse_args().__dict__)
        return self

    def train(self):
        dataset = CustomDataset(self.data_file, window_size=self.window_size)
        dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

        self.model.train()
        for epoch_index in range(self.epoch):
            tbar = tqdm(total=len(dataloader), desc=f"Epoch {epoch_index}")
            for batch_index, [data, gold] in enumerate(dataloader):
                self.optimizer.zero_grad()

                prediction = self.model(data)
                loss = functional.cross_entropy(prediction, gold.cuda())
                loss.backward()
                self.optimizer.step()

                tbar.set_postfix(loss=loss.item())
                tbar.update()
            tbar.close()

            with torch.no_grad():
                predicts, golds = [], []
                eval_dataset = CustomDataset('eval', window_size=self.window_size)
                eval_dataloader = DataLoader(eval_dataset, batch_size=128, shuffle=False, drop_last=False)
                for batch_index, [data, gold] in enumerate(eval_dataloader):
                    predicts += self.model(data).argmax(dim=1).tolist()
                    golds += gold.tolist()
            print(metrics.classification_report(golds, predicts))

        torch.save(self.model.state_dict(), self.out_model_file)

    def create_optimizer(self):
        return torch.optim.AdamW(self.model.parameters(), lr=1e-4)

    @staticmethod
    def model_init():
        return Model().cuda()

    def __call__(self):
        return self.train()


if __name__ == '__main__':
    trainer = Trainer(epochs=50, window_size=10)
    trainer.parse()
    trainer()
