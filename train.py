import torch
from torch.nn import functional
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from algo import Model
from utils.dataset import MyDataset
import argparse
from sklearn import metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, default='data')
    parser.add_argument('--out_model_file', type=str, default='out_model_file')
    parser.add_argument('--in_model_file', type=str, default='None')
    args = parser.parse_args()
    window_size = 10
    dataset = MyDataset(args.data_file, mode='train', window_size=window_size)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    model = Model().cuda()

    epoch = 10
    best = 0
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    for epoch_index in range(epoch):
        tbar = tqdm(total=len(dataloader), desc=f"Epoch {epoch_index}")
        for batch_index, [data, gold] in enumerate(dataloader):
            optimizer.zero_grad()

            prediction = model(data)
            loss = functional.cross_entropy(prediction, gold.cuda())
            loss.backward()
            optimizer.step()

            tbar.set_postfix(loss=loss.item())
            tbar.update()
        tbar.close()
    torch.save(model.state_dict(), args.out_model_file)

    predicts, golds = [], []
    with torch.no_grad():
        eval_dataset = MyDataset('eval', mode='test', window_size=window_size)
        eva_dataloader = DataLoader(eval_dataset, batch_size=128, shuffle=False)
        for batch_index, [data, gold] in enumerate(dataloader):
            predicts += model(data).argmax(dim=1).tolist()
            golds += gold.tolist()
    print(metrics.classification_report(golds, predicts))