from sklearn import ensemble

from MLBase import Predictor
from utils import PredictParser

if __name__ == '__main__':
    parser = PredictParser()
    args = parser.parse_args()
    model = ensemble.RandomForestClassifier()
    predictor = Predictor(model, **vars(args))
    predictor()
