import argparse

from utils import Predictor

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('in_model_file', type=str, default='parameters.pt')
    args = parser.parse_args()

    predictor = Predictor(**vars(args))
    predictor()
