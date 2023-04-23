from fastai.vision.all import *
from pynvml import *
from tqdm import tqdm
from PIL import Image
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

#Dataset to predict on
fnames = get_image_files(path/'datasets/18102016_Part01')
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

sz = np.array([288,352])
print(sz)

size = sz

handle = nvmlDeviceGetHandleByIndex(0)
info = nvmlDeviceGetMemoryInfo(handle)
print("Free Memory: ", info.free/1048576)

free = info.free/1048576
# the max size of bs depends on the available GPU RAM
if free > 8200: bs=8
else:           bs=4
print(f"using bs={bs}, have {free}MB of GPU RAM free")

half = tuple(int(x/2) for x in sz)
print(half)

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

#Model
opt = ranger
learn = learn = unet_learner(dls, resnet34, self_attention=True, act_cls=Mish, opt_func=opt)
learn.load('stage-2-fullsize-weights')

results_save = 'datasets/18102016_Part01_results_updated'
path_rst = path/results_save
path_rst.mkdir(exist_ok=True)

dl = learn.dls.test_dl(fnames)
preds = learn.get_preds(dl=dl)

for i, pred in enumerate(preds[0]):
    pred_arg = pred.argmax(dim=0)
    torch.save(pred_arg, path_rst/f'Image_{i}.pt')

