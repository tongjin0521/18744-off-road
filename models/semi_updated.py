from fastai.vision.all import *
from pynvml import *
from pathlib import Path
import torch
torch.backends.cudnn.benchmark=True
import numpy as np
from tqdm import tqdm
import warnings
import shutil
import cv2 as cv
cv.__version__

nvmlInit()
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")

'''
Preparing the Data
'''
# Set path variables
path = Path('../')

codes = np.loadtxt(path/'codes.txt', dtype=str)
path_img = path/'images'
path_img_unlabled = path/'original_unlabled_images'
path_lbl = path/'labels'

fnames = get_image_files(path_img)
unlabeled_names = get_image_files(path_img_unlabled)
lbl_names = get_image_files(path_lbl)

get_y_fn = lambda x: path_lbl/f'{x.stem}{x.suffix}'

size = np.array([288,352])

handle = nvmlDeviceGetHandleByIndex(0)
info = nvmlDeviceGetMemoryInfo(handle)
print("Free Memory: ", info.free/1048576)

free = info.free/1048576
# the max size of bs depends on the available GPU RAM
if free > 16400: bs=16
else:           bs=4
print(f"using bs={bs}, have {free}MB of GPU RAM free")

def FileSplitter(fname):
    "Split `items` depending on the value of `mask`."
    valid = Path(fname).read_text().split('\n') 
    def _func(x): return x.name in valid
    def _inner(o, **kwargs): return FuncSplitter(_func)(o)
    return _inner

#Datasets
data = DataBlock(blocks=(ImageBlock, MaskBlock(codes)),
                   get_items=get_image_files,
                   splitter=FileSplitter(path/'valid.txt'),
                   get_y=get_y_fn,
                   batch_tfms=[*aug_transforms(size=size), Normalize.from_stats(*imagenet_stats)])

dls = data.dataloaders(path_img, bs=bs, num_workers=0)

#Model
opt = ranger
learn = learn = unet_learner(dls, resnet34, self_attention=True, act_cls=Mish, opt_func=opt)
learn.load('stage-2-fullsize-weights')

#Create Mask Function
def colorfull_fast(frame):
  # grab the image dimensions
  width = 288
  height = 352
  image = np.empty((width, height, 3), dtype=np.uint8)
    
  # loop over the image, pixel by pixel
  for x in range(width):
    for y in range(height):
      pixel = frame[x, y]
      if pixel == 0: #background
        image[x, y] = (0,0,0)
      elif pixel == 1: #roadAsphalt
        image[x, y] = (1,1,1)
      elif pixel == 2: #roadPaved
        image[x, y] = (2,2,2)
      elif pixel == 3: #roadUnpaved
        image[x, y] = (3,3,3)
      elif pixel == 4: #roadMarking
        image[x, y] = (4,4,4)
      elif pixel == 5: #speedBump
        image[x, y] = (5,5,5)
      elif pixel == 6: #catsEye
        image[x, y] = (6,6,6)        
      elif pixel == 7: #stormDrain
        image[x, y] = (7,7,7)
      elif pixel == 8: #manholeCover
        image[x, y] = (8,8,8)
      elif pixel == 9: #patchs
        image[x, y] = (9,9,9)
      elif pixel == 10: #waterPuddle
        image[x, y] = (10,10,10)
      elif pixel == 11: #pothole
        image[x, y] = (11,11,11)
      elif pixel == 12: #cracks
        image[x, y] = (12,12,12)
  
  # return the colored image
  return image

#Create test directory
# path_rst = path/'labelstest'
# path_rst.mkdir(exist_ok=True)

#Predict image
num_next_round_imgs = 0

dl = learn.dls.test_dl(unlabeled_names)
preds = learn.get_preds(dl=dl)

for i, pred in enumerate(preds[0]):
    img_s = unlabeled_names[i]
    img_split = f'{img_s}'
    img_split = "1_" + img_split.split("\\")[-1]
    
    pred_prob = torch.max(pred, dim=0)[0]
    mean_prob = torch.mean(pred_prob)
    min_prob = torch.min(pred_prob)

    if (mean_prob > 0.98 and min_prob > 0.2):
        shutil.copyfile(f'{img_s}', "../images/" + img_split)
        num_next_round_imgs += 1
        
        #Create the label
        pred_arg = pred.argmax(dim=0).numpy()
        pred_arg = colorfull_fast(pred_arg)

        im = Image.fromarray(pred_arg)
        im.save(path_lbl/img_split)

#Print number of added images
print(num_next_round_imgs)

