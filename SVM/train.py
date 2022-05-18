from utils import TrainParser
from MLBase.trainer import Trainer
import sklearn

if __name__ == '__main__':
    parser = TrainParser()
    args = parser.parse_args()
    model = sklearn.svm.SVC(gamma='scale')
    trainer = Trainer(model, **vars(args))
    trainer()