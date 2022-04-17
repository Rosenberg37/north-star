from torch import Tensor, nn


class Model(nn.Module):
    def __init__(self, hidden_size: int = 256, num_layers: int = 1):
        super(Model, self).__init__()
        self.transform = nn.Sequential(
            nn.Linear(90, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, hidden_size)
        )
        self.rnn = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.predict = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 6)
        )

    def forward(self, inputs: Tensor):
        rnn_hidden = self.rnn(self.transform(inputs))
        return self.predict(rnn_hidden[1][0])
