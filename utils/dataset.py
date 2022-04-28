import os

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


class CustomDataset(Dataset):
    label2idx = {
        'sit': 0,
        'stand': 1,
        'walk': 2,
        'upstairs': 3,
        'downstairs': 4,
        'run': 5
    }
    idx2label = dict(zip(label2idx.values(), label2idx.keys()))

    def __init__(self, data_directory_path: str, window_size: int = 20):
        self.window = window_size

        data_paths = []
        if os.path.exists(data_directory_path):
            data_paths = os.listdir(data_directory_path)
            data_paths = list(filter(lambda p: p.endswith('.csv') and self.get_motion(p), data_paths))
            data_paths = list(map(lambda p: os.path.join(data_directory_path, p), data_paths))

        self.samples, self.golds = [], []
        for path in tqdm(data_paths, desc='Reading data'):
            idx = self.label2idx[self.get_motion(path)]
            sample = torch.as_tensor(pd.read_csv(path, header=None).values, dtype=torch.float)
            self.samples.append(sample.cuda())
            self.golds.append(idx)
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

    def get_motion(self, pathname: str):
        for name in list(self.label2idx.keys()):
            if name in pathname:
                return name
        return None


if __name__ == '__main__':
    dataset = CustomDataset('../data')
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    print(next(iter(dataloader)))
