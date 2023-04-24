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
print(path.ls())

codes = np.loadtxt(path/'codes.txt', dtype=str)
path_img = path/'images'
path_lbl = path/'labels'

fnames = get_image_files(path_img)
print(fnames[:3])
print(len(fnames))

lbl_names = get_image_files(path_lbl)

img_f = fnames[139]
img = PILImage.create(img_f)
#img.show(figsize=(5,5))
# plt.show()

get_y_fn = lambda x: path_lbl/f'{x.stem}{x.suffix}'

mask = PILMask.create(get_y_fn(img_f))
#mask.show(figsize=(5,5), alpha=1)
# plt.show()

sz = mask.shape
print(sz)

handle = nvmlDeviceGetHandleByIndex(0)
info = nvmlDeviceGetMemoryInfo(handle)
print("Free Memory: ", info.free/1048576)

free = info.free/1048576
# the max size of bs depends on the available GPU RAM
if free > 16400: bs=16
else:           bs=4
print(f"using bs={bs}, have {free}MB of GPU RAM free")

#Takes in filenames and checks for filenames in validation filenames
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
                   batch_tfms=[*aug_transforms(size=sz), Normalize.from_stats(*imagenet_stats)])

dls = data.dataloaders(path_img, bs=bs, num_workers=0)

# dls.show_batch(max_n=4, vmin=1, vmax=30, figsize=(14,10), show=True)
# plt.show()

dls.vocab = codes

name2id = {v:k for k,v in enumerate(codes)}
print(name2id)

#Accuracy Function
void_code = name2id['manholeCover']
def acc_test(inp, targ):
    targ = targ.squeeze(1)
    mask = targ != void_code
    return (inp.argmax(dim=1)[mask]==targ[mask]).float().mean()

#Model
opt = ranger
balanced_loss = CrossEntropyLossFlat(axis=1, weight=torch.tensor([1.0,5.0,6.0,7.0,75.0,1000.0,3100.0,3300.0,0.0,270.0,2200.0,1000.0,180.0]).cuda())
learn = unet_learner(dls, resnet34, metrics=acc_test, self_attention=True, act_cls=Mish, opt_func=opt, loss_func=balanced_loss)
# learn.load('stage-2-fullsize-more')

# lr = learn.lr_find()
# print(lr)
# plt.show()

#TODO: increase this to lower loss faster
lr = 1e-5

# learn.unfreeze()
# lrs = slice(lr/400,lr/4)

# learn.fit_flat_cos(20, lrs, pct_start=0.72)
# learn.save('stage-2-fullsize-weights')
# learn.show_results(max_n=4, figsize=(15,15))
# plt.show()

'''
Continue Training - 400 epochs currently
'''
learn.load('stage-2-fullsize-weights')
# learn.unfreeze()
# lrs = slice(lr/400,lr/4)
# learn.fit_flat_cos(50, lrs, pct_start=0.72)
# learn.save('stage-2-fullsize-weights')
learn.show_results(max_n=4, figsize=(15,15))
plt.show()

# #Interpret
# interp = SegmentationInterpretation.from_learner(learn)
# top_losses, top_idxs = interp.top_losses()

# plt.hist(to_np(top_losses), bins=20)
# interp.plot_top_losses(10)
# plt.savefig('with weights top losses.png')

# plt.show()

# print(interp)

# interp.plot_confusion_matrix()

# plt.show()

