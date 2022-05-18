from sklearn import neighbors

from MLBase.trainer import Trainer
from utils import TrainParser

if __name__ == '__main__':
    parser = TrainParser()
    args = parser.parse_args()
    model = neighbors.KNeighborsClassifier()
    trainer = Trainer(model, **vars(args))
    trainer()
