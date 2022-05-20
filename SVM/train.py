from utils import TrainParser
from MLBase.trainer import Trainer
import sklearn
from sklearn.linear_model import SGDClassifier

if __name__ == '__main__':
    parser = TrainParser()
    args = parser.parse_args()
    # model = SGDClassifier(loss="hinge")
    model = sklearn.svm.SVC(gamma='scale')
    trainer = Trainer(model, **vars(args))
    trainer()