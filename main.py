import torch
from torch.nn import functional
from torch.utils.data import DataLoader
from tqdm import tqdm

from algo import Model
from utils.dataset import MyDataset

if __name__ == '__main__':
    window_size = 20
    dataset = MyDataset('train', window_size)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    model = Model().cuda()

    epoch = 100
    best = 0
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    for epoch_index in range(epoch):
        tbar = tqdm(total=len(dataloader), desc=f"Epoch {epoch_index}")
        for batch_index, [data, gold] in enumerate(dataloader):
            optimizer.zero_grad()

            prediction = model(data)
            loss = functional.cross_entropy(prediction, gold)
            loss.backward()
            optimizer.step()

            tbar.set_postfix(loss=loss.item())
            tbar.update()
        tbar.close()

        if (epoch_index + 1) % 10 == 0:
            model.eval()
            TP, FP = 0, 0
            for data, golden in tqdm(MyDataset('test', window_size)):
                prediction = torch.argmax(model(data.unsqueeze(0)))
                if prediction == golden:
                    TP += 1
                else:
                    FP += 1
            f1 = TP / (TP + FP)
            print(f1)
            if f1 > best:
                torch.save(model.state_dict(), 'cache/params.pt')
                best = f1
            model.train()
