import argparse
import codecs
import sys

from utils import Predictor

if __name__ == "__main__":
    sys.stdin = codecs.open("whatever.csv", 'r', encoding='utf-8')
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_model_file', type=str, default='out_model_file')
    args = parser.parse_args()

    predictor = Predictor(**vars(args))
    predictor()
