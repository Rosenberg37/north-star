import os

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


class CustomDataset(Dataset):
    def __init__(self, pathname: str, window_size: int = 10, mode: str = 'train'):
        self.window = window_size

        path_list = []
        for dir_path, dir_names, filenames in os.walk(pathname):
            for filename in filenames:
                path = os.path.join(dir_path, filename)
                if path.endswith('.xlsx') and CustomDataset.get_motion(path) is not None:
                    path_list.append(path)
        label2id, id2label = CustomDataset.convert2id()
        self.samples, self.golds = [], []
        for path in path_list:
            label = CustomDataset.get_motion(path)
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

    @staticmethod
    def get_motion(pathname: str):
        motion = ['sit', 'downstairs', 'upstairs', 'stand', 'run', 'walk']
        for i in motion:
            if i in pathname:
                return i
        return None

    @staticmethod
    def convert2id():
        label2id = {'sit': 0, 'downstairs': 4, 'upstairs': 3, 'stand': 1, 'run': 5, 'walk': 2}
        id2label = {0: 'sit', 1: 'stand', 2: 'walk', 3: 'upstairs', 4: 'downstairs', 5: 'run'}
        return label2id, id2label


if __name__ == '__main__':
    dataset = CustomDataset('../data')
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    print(next(iter(dataloader)))
