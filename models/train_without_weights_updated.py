from fastai.vision.all import *
import matplotlib.pyplot as plt
from pynvml import *
nvmlInit()

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")

'''
Preparing the Data
'''
# Set path variables
path = Path('../')

path_img = path/'images'
path_lbl = path/'labels'

fnames = get_image_files(path_img)

lbl_names = get_image_files(path_lbl)

img_f = fnames[139]
img = PILImage.create(img_f)

get_y_fn = lambda x: path_lbl/f'{x.stem}{x.suffix}'

mask = PILMask.create(get_y_fn(img_f))

codes = np.loadtxt(path/'codes.txt', dtype=str)

#Takes in filenames and checks for filenames in validation filenames
def FileSplitter(fname):
    "Split `items` depending on the value of `mask`."
    valid = Path(fname).read_text().split('\n') 
    def _func(x): return x.name in valid
    def _inner(o, **kwargs): return FuncSplitter(_func)(o)
    return _inner

'''
First round, will train at half the image size
'''
sz = mask.shape

half = tuple(int(x/2) for x in sz)

#Determine batch size based on avaiable GPU mem
handle = nvmlDeviceGetHandleByIndex(0)
info = nvmlDeviceGetMemoryInfo(handle)
print("Free Memory: ", info.free/1048576)

free = info.free/1048576
# the max size of bs depends on the available GPU RAM
if free > 16400: bs=16
else:           bs=4
print(f"using bs={bs}, have {free}MB of GPU RAM free")

'''
First Step - Without Weights (half size images)
'''
# Datasets
# data = DataBlock(blocks=(ImageBlock, MaskBlock(codes)),
#                    get_items=get_image_files,
#                    splitter=FileSplitter(path/'valid.txt'),
#                    get_y=get_y_fn,
#                    batch_tfms=[*aug_transforms(size=half), Normalize.from_stats(*imagenet_stats)])

# dls = data.dataloaders(path_img, bs=bs, num_workers=0)

# dls.vocab = codes

name2id = {v:k for k,v in enumerate(codes)}

#Accuracy Function
void_code = name2id['manholeCover']
def acc_test(inp, targ):
    targ = targ.squeeze(1)
    mask = targ != void_code
    return (inp.argmax(dim=1)[mask]==targ[mask]).float().mean()

#Model
# opt = ranger
# learn = unet_learner(dls, resnet34, metrics=acc_test, self_attention=True, act_cls=Mish, opt_func=opt)

# lr = learn.lr_find(suggest_funcs=(valley, slide, minimum))
# print(lr)
# plt.show()

# lr=9.12e-5

# learn.fit_flat_cos(10, slice(lr), pct_start=0.72)
# learn.save('semi-1-plus-stage-1')
# # learn.show_results(max_n=4, figsize=(15,15))
# # plt.show()

# learn.load('semi-1-plus-stage-1')
# lrs = slice(lr/400,lr/4)
# learn.unfreeze()
# learn.fit_flat_cos(12, lrs, pct_start=0.72)
# learn.save('semi-1-plus-stage-2')
# learn.show_results(max_n=4, figsize=(15,15))
# plt.show()

'''
Train with full size images
'''
data = DataBlock(blocks=(ImageBlock, MaskBlock(codes)),
                   get_items=get_image_files,
                   splitter=FileSplitter(path/'valid.txt'),
                   get_y=get_y_fn,
                   batch_tfms=[*aug_transforms(size=sz), Normalize.from_stats(*imagenet_stats)])

dls = data.dataloaders(path_img, bs=bs, num_workers=0)

dls.vocab = codes

opt = ranger
learn = unet_learner(dls, resnet34, metrics=acc_test, self_attention=True, act_cls=Mish, opt_func=opt)
learn.load('semi-1-plus-stage-2')

# lr = learn.lr_find(suggest_funcs=(valley, slide, minimum))
# print(lr)
# plt.show()

lr=2.75e-4

learn.fit_flat_cos(10, slice(lr), pct_start=0.72)
learn.save('semi-1-plus-stage-2-fullsize')

learn.load('semi-1-plus-stage-2-fullsize')
learn.unfreeze()
lrs = slice(lr/400,lr/4)
learn.fit_flat_cos(20, lrs, pct_start=0.72)
learn.save('semi-1-plus-stage-2-fullsize')
# learn.show_results(max_n=4, figsize=(15,15))
# plt.show()