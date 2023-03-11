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
import shutil
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")

# prepare the path
path = Path('../')
codes = np.loadtxt(path/'codes.txt', dtype=str)
path_img = path/'images'
path_img_unlabled = path/'images_unlabled_Part01'
path_lbl = path/'labels'

fnames = get_image_files(path_img)
unlabled_names = get_image_files(path_img_unlabled)
lbl_names = get_image_files(path_lbl)

get_y_fn = lambda x: path_lbl/f'{x.stem}{x.suffix}'

size = np.array([288,352])

free = gpu_mem_get_free_no_cache()
# the max size of bs depends on the available GPU RAM
if free > 8200: bs=8
else:           bs=4
print(f"using bs={bs}, have {free}MB of GPU RAM free")

#load model
src = (SegmentationItemList.from_folder(path_img)
       .split_by_fname_file('../valid.txt')
       .label_from_func(get_y_fn, classes=codes))
data = (src.transform(get_transforms(), size=size, tfm_y=True)
        .databunch(bs=bs)
        .normalize(imagenet_stats))

learn = unet_learner(data, models.resnet34)
learn.load('stage-2-weights')

#create test directory
results_save = 'results_test'
path_rst = path/results_save
path_rst.mkdir(exist_ok=True)

#predict image
num_next_round_imgs = 0
for unlabled_name_i in tqdm(unlabled_names):
        img_s =unlabled_name_i
        img_toSave = open_image(img_s)
        img_split = f'{img_s}'
        img_split = "1_"+img_split.split("/")[-1]
        predictionSave = learn.predict(img_toSave)
        pred_prob = predictionSave[2]
        pred_prob = torch.max(pred_prob,dim=0)[0]
        # print(torch.mean(pred_prob),torch.min(pred_prob),torch.median(pred_prob))
        mean_prob = torch.mean(pred_prob)
        min_prob = torch.min(pred_prob)
        #TODO: how to select next round labeled data
        if (mean_prob > 0.975 and min_prob > 0.21):
                # print(num_next_round_imgs)
                shutil.copyfile(f'{img_s}', "../images/"+img_split)
                num_next_round_imgs+=1
                predictionSave[0].save(path_rst/img_split) #Save Image
                break

print(num_next_round_imgs)
