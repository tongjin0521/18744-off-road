from fastai.vision.all import *
from pynvml import *
from tqdm import tqdm
nvmlInit()

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")

'''
Preparing the Data
'''
# Set path variables
path = Path('../')

codes = np.loadtxt(path/'codes.txt', dtype=str)
path_img = path/'images'
path_lbl = path/'labels'

# 18102016_Part01
# 14042017_Part06

#Dataset to predict on
fnames = get_image_files(path/'datasets/14042017_Part06')

lbl_names = get_image_files(path_lbl)

img_f = fnames[139]
img = PILImage.create(img_f)

get_y_fn = lambda x: path_lbl/f'{x.stem}{x.suffix}'

sz = np.array([288,352])

size = sz

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
                   batch_tfms=[*aug_transforms(size=sz), Normalize.from_stats(*imagenet_stats)])

dls = data.dataloaders(path_img, bs=bs, num_workers=0)

#Model
opt = ranger
learn = learn = unet_learner(dls, resnet34, self_attention=True, act_cls=Mish, opt_func=opt)
learn.load('pre-semi-stage-2-fullsize')

results_save = 'datasets/14042017_Part06_pre_semi_weights'
path_rst = path/results_save
path_rst.mkdir(exist_ok=True)

dl = learn.dls.test_dl(fnames)
preds = learn.get_preds(dl=dl)

for i, pred in tqdm(enumerate(preds[0])):
    pred_arg = pred.argmax(dim=0)
    torch.save(pred_arg, path_rst/f'Image_{i}.pt')

