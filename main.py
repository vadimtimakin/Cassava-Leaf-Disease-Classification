from torch.cuda import is_available, empty_cache
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"   # RTX 3080 with cuda-11 fix
print(is_available())
empty_cache()

from config import Cfg
from utils import fullseed
from train_functions import run

import warnings
warnings.filterwarnings("ignore")

fullseed(Cfg.seed)
run(Cfg)