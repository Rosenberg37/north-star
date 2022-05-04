import argparse

from utils import Trainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("data_file", nargs='?', type=str, default='data')
    parser.add_argument('out_model_file', nargs='?', type=str, default='parameters.pt')
    parser.add_argument('in_model_file', nargs='?', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--debug', action='store_true', default=False)
    args = parser.parse_args()

    trainer = Trainer(**vars(args))
    trainer()
