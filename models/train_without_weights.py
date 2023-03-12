from fastai.vision import *
from fastai.vision.interpret import *
from fastai.callbacks.hooks import *
from pathlib import Path
from fastai.utils.mem import *
import torch
torch.backends.cudnn.benchmark=True
import numpy as np

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

name2id = {v:k for k,v in enumerate(codes)}

def acc_rtk(input, target):
    target = target.squeeze(1)
    mask = target != 0
    return (input.argmax(dim=1)[mask]==target[mask]).float().mean()
    
metrics=acc_rtk
wd=1e0

learn = unet_learner(data, models.resnet34, metrics=metrics, wd=wd)

lr_find(learn)
learn.recorder.plot()
plt.savefig('loss_plot.png')

# lr is set based on the fig above
lr=1e-3
#TODO: # of iterations = 10 (previously)
learn.fit_one_cycle(10, slice(lr), pct_start=0.9)
learn.save('stage-1')
learn.load('stage-1')
learn.show_results(rows=5, figsize=(15,15))

#training without weights
learn.unfreeze()
lrs = slice(lr/400,lr/4)

#TODO: # of iterations = 100 (previously)
learn.fit_one_cycle(100, lrs, pct_start=0.9)
learn.save('stage-2')

interp = SegmentationInterpretation.from_learner(learn)
top_losses, top_idxs = interp.top_losses((288,352))
mean_cm, single_img_cm = interp._generate_confusion()
df = interp._plot_intersect_cm(mean_cm, "Mean of Ratio of Intersection given True Label")
plt.savefig('confusion.png')
