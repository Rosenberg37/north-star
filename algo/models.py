from torch import Tensor, nn


class Model(nn.Module):
    def __init__(self, input_size: int = 90, hidden_size: int = 128, num_layers: int = 1, num_labels: int = 6):
        super(Model, self).__init__()
        self.transform = nn.Linear(input_size, hidden_size)
        self.rnn = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.predict = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, num_labels)
        )

    def forward(self, inputs: Tensor) -> Tensor:
        rnn_hidden = self.rnn(self.transform(inputs))
        return self.predict(rnn_hidden[1][0])
