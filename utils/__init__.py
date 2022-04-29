from .dataset import *
from .predictor import *
from .trainer import *

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
window_size: int = 40

# logger
logger = logging.getLogger("north_star")
logger.setLevel(logging.INFO)
log_format = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")

file_handler = logging.FileHandler(f"runtime.log")
file_handler.setFormatter(log_format)
logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setFormatter(log_format)
logger.addHandler(console_handler)
