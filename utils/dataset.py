import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import os

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class MyDataset(Dataset):
    def __init__(self, pathname, mode='train', window_size=10):
        self.window = window_size
        pathlst = []
        for dirpath, dirnames, filenames in os.walk(pathname):
            for filename in filenames:
                path = os.path.join(dirpath, filename)
                if path.endswith('.xlsx') and MyDataset.get_motion(path) is not None:
                    pathlst.append(path)
        label2id, id2label = MyDataset.convert2id()
        self.samples, self.golds = [], []
        for path in pathlst:
            label = MyDataset.get_motion(path)
            id = label2id[label]
            sample = torch.tensor(list(pd.read_excel(path, header=None).values)).float().to(device)
            self.samples.append(sample)
            self.golds.append(id)
        self.length = [0] + [sample.shape[0] - self.window + 1 for sample in self.samples]
        for i in range(1, len(self.length)):
            self.length[i] += self.length[i - 1]

    def get_sample_index(self, idx):
        l, r = 0, len(self.length)
        while l + 1 < r:
            m = (l + r) // 2
            if idx < self.length[m]:
                r = m
            else:
                l = m
        assert l < len(self.length)
        return l

    def __len__(self):
        return self.length[-1]

    def __getitem__(self, index: int):
        sample_index = self.get_sample_index(index)
        assert sample_index >= 0
        start = 0 if sample_index == 0 else index - self.length[sample_index]
        sample = self.samples[sample_index][start: start + self.window]
        gold = self.golds[sample_index]
        return sample, gold

    def get_motion(pathname: str):
        motion = ['sit', 'downstair', 'upstair', 'stand', 'run', 'walk']
        for i in motion:
            if i in pathname:
                return i
        return None

    def convert2id():
        label2id = {'sit': 0, 'downstair': 4, 'upstair': 3, 'stand': 1, 'run': 5, 'walk':2}
        id2label = {0: 'sit', 1: 'stand', 2: 'walk', 3: 'upstair', 4: 'downstair', 5: 'run'}
        return label2id, id2label

if __name__ == '__main__':
    dataset = MyDataset('../data')
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    print(next(iter(dataloader)))