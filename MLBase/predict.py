import sklearn

from MLBase import Predictor
from utils import PredictParser

if __name__ == '__main__':
    parser = PredictParser()
    args = parser.parse_args()
    model = sklearn.svm.SVC(gamma='scale')
    predictor = Predictor(model, **vars(args))
    predictor()
