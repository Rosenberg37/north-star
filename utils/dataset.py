import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, mode: str, window_size: int):
        if mode == 'train':
            self.data = torch.Tensor(pd.read_csv('raw_data/train.csv', header=None).values)
            self.golden = torch.Tensor(pd.read_csv('raw_data/train_golden.csv', header=None).values).squeeze(-1)
        elif mode == 'test':
            self.data = torch.Tensor(pd.read_csv('raw_data/test.csv', header=None).values)
            self.golden = torch.Tensor(pd.read_csv('raw_data/test_golden.csv', header=None).values).squeeze(-1)

        self.data = self.data.view(-1, window_size, 90)
        self.golden = self.golden.long()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        data = self.data[index].cuda()
        golden = self.golden[index].cuda()
        return data, golden
