import os
import glob
import base64
from matplotlib import pyplot as plt
import cv2 as cv
cv.__version__
import torch
from PIL import Image

from pathlib import Path
from tqdm import tqdm

colored_results = 'datasets/18102016_Part01_results_updated_color'
path = Path('../')
path_crst = path/colored_results
path_crst.mkdir(exist_ok=True)
results_save = 'datasets/18102016_Part01_results_updated'
path_rst = path/results_save

import cv2 as cv
import numpy as np


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
        image[x, y] = (85,85,255)
      elif pixel == 2: #roadPaved
        image[x, y] = (85,170,127)
      elif pixel == 3: #roadUnpaved
        image[x, y] = (255,170,127) 
      elif pixel == 4: #roadMarking
        image[x, y] = (255,255,255) 
      elif pixel == 5: #speedBump
        image[x, y] = (255,85,255)
      elif pixel == 6: #catsEye
        image[x, y] = (255,255,127)          
      elif pixel == 7: #stormDrain
        image[x, y] = (170,0,127) 
      elif pixel == 8: #manholeCover
        image[x, y] = (0,255,255) 
      elif pixel == 9: #patchs
        image[x, y] = (0,0,127) 
      elif pixel == 10: #waterPuddle
        image[x, y] = (170,0,0)
      elif pixel == 11: #pothole
        image[x, y] = (255,0,0)
      elif pixel == 12: #cracks
        image[x, y] = (255,85,0)
  
  image = cv.cvtColor(image,cv.COLOR_BGR2RGB)
  
  # return the colored image
  return image

# path_rst = path/'results_test'
filenames = [img for img in glob.glob(str(path_rst/"*.pt"))]

filenames.sort() # ADD THIS LINE

for tensor in tqdm(filenames):
  frame = torch.load(tensor).numpy()
  
  #%timeit colorfull_fast(frame)
  
  frame = colorfull_fast(frame)
  name = f'{tensor}'.split("\\")[-1]
  name = os.path.splitext(name)[0]+'.png'
  print(name)
 
  im = Image.fromarray(frame)
  im.save(os.path.join(path_crst, name))
  cv.imwrite(os.path.join(path_crst, name), frame)

# frame = cv.imread(filenames[0])
#print(frame)
  
#%timeit colorfull_fast(frame)

# frame = torch.load(filenames[0]).numpy()
  
# frame = colorfull_fast(frame)
# plt.imshow(frame)
# plt.show()

# name = f'{filenames[0]}'.split("\\")[-1]
# print(name)
# res = cv.imwrite(os.path.join(path_crst, name), frame)

# print(os.path.join(path_crst, name))
print("Done!")