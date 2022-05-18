from sklearn import tree

from MLBase import Predictor
from utils import PredictParser

if __name__ == '__main__':
    parser = PredictParser()
    args = parser.parse_args()
    model = tree.DecisionTreeClassifier()
    predictor = Predictor(model, **vars(args))
    predictor()
