from sklearn import svm

from MLBase.trainer import Trainer
from utils import TrainParser

if __name__ == '__main__':
    parser = TrainParser()
    args = parser.parse_args()
    model = svm.SVC(gamma='scale')
    trainer = Trainer(model, **vars(args))
    trainer()
