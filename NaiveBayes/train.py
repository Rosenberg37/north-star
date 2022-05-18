from sklearn import naive_bayes

from MLBase.trainer import Trainer
from utils import TrainParser

if __name__ == '__main__':
    parser = TrainParser()
    args = parser.parse_args()
    model = naive_bayes.GaussianNB()
    trainer = Trainer(model, **vars(args))
    trainer()
