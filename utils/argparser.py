import argparse


class TrainParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("data_file", nargs='?', type=str, default='../data')
        self.parser.add_argument('out_model_file', nargs='?', type=str, default='parameters.pt')
        self.parser.add_argument('in_model_file', nargs='?', type=str, default=None)
        self.parser.add_argument('--epochs', type=int, default=20)
        self.parser.add_argument('--debug', action='store_true', default=True)

    def parse_args(self, *args, **kwargs):
        return self.parser.parse_args()


class PredictParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('in_model_file', nargs='?', type=str, default='parameters.pt')

    def parse_args(self):
        return self.parser.parse_args()
