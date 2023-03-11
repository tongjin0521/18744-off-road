from fastai.vision import *
from fastai.vision.interpret import *
from fastai.callbacks.hooks import *
from pathlib import Path
from fastai.utils.mem import *
torch.backends.cudnn.benchmark=True

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")

path = Path('gdrive/My Drive/Colab Notebooks/data/')
path.ls()