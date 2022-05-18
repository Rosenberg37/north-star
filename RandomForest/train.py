from sklearn import ensemble

from MLBase.trainer import Trainer
from utils import TrainParser

if __name__ == '__main__':
    parser = TrainParser()
    args = parser.parse_args()
    model = ensemble.RandomForestClassifier()
    trainer = Trainer(model, **vars(args))
    trainer()
