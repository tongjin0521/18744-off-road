from fastai.vision import *
from fastai.vision.interpret import *
from fastai.callbacks.hooks import *
from pathlib import Path
from fastai.utils.mem import *
import torch
torch.backends.cudnn.benchmark=True
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")

# prepare the path
path = Path('../')
codes = np.loadtxt(path/'codes.txt', dtype=str)
path_img = path/'images'
path_lbl = path/'labels'

fnames = get_image_files(path_img)
lbl_names = get_image_files(path_lbl)

img_f = fnames[139]

get_y_fn = lambda x: path_lbl/f'{x.stem}{x.suffix}'

mask = open_mask(get_y_fn(img_f))
src_size = np.array(mask.shape[1:])

size = src_size

free = gpu_mem_get_free_no_cache()
# the max size of bs depends on the available GPU RAM
if free > 8200: bs=8
else:           bs=4
print(f"using bs={bs}, have {free}MB of GPU RAM free")

src = (SegmentationItemList.from_folder(path_img)
       .split_by_fname_file('../valid.txt')
       .label_from_func(get_y_fn, classes=codes))
data = (src.transform(get_transforms(), size=size, tfm_y=True)
        .databunch(bs=bs)
        .normalize(imagenet_stats))

learn = unet_learner(data, models.resnet34)
learn.load('stage-2-weights')

results_save = 'results'
path_rst = path/results_save
path_rst.mkdir(exist_ok=True)

def save_preds(names):
    #names = dl.dataset.items
    
    for fname_i in tqdm(names):
        img_s = fname_i
        img_toSave = open_image(img_s)
        img_split = f'{img_s}'
        img_split = img_split.split("/")[-1]
        predictionSave = learn.predict(img_toSave)
        predictionSave[0].save(path_rst/img_split) #Save Image
        
save_preds(fnames)
