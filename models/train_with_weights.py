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
void_code = name2id['manholeCover']

def acc_rtk(input, target):
    target = target.squeeze(1)
    mask = target != void_code
    return (input.argmax(dim=1)[mask]==target[mask]).float().mean()
    
metrics=acc_rtk
wd=1e0

balanced_loss = CrossEntropyFlat(axis=1, weight=torch.tensor([1.0,5.0,6.0,7.0,75.0,1000.0,3100.0,3300.0,0.0,270.0,2200.0,1000.0,180.0]).cuda())
learn = unet_learner(data, models.resnet34, metrics=metrics, loss_func=balanced_loss, wd=wd)
learn.load('stage-2')
#CUDA_LAUNCH_BLOCKING=1

# lr_find(learn)
# learn.recorder.plot()
# plt.savefig('loss_plot_with_weights.png')

lr=1e-5
learn.unfreeze()
lrs = slice(lr/400,lr/4)
#TODO: Change # of steps
learn.fit_one_cycle(100, lrs, pct_start=0.8)
# learn.save('stage-2-weights')

interp = SegmentationInterpretation.from_learner(learn)
mean_cm, single_img_cm = interp._generate_confusion()
# global class performance
df = interp._plot_intersect_cm(mean_cm, "Mean of Ratio of Intersection given True Label")
# single image class performance
plt.savefig('global_confusion_with_weights.png')
i = 130
df = interp._plot_intersect_cm(single_img_cm[i], f"Ratio of Intersection given True Label, Image:{i}")

# show xyz
interp.show_xyz(i)

learn.interpret
