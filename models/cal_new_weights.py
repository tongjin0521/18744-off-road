import os
import glob
import base64
import cv2 as cv
import cv2
cv.__version__

import multiprocessing
from pathlib import Path
from tqdm import tqdm
import numpy as np




path = Path('../')
masks_folder = 'test_mask'
path_rst = path/masks_folder

width = 288
height = 352


def process_image(path, counts):
    img = cv.imread(path)
    # grab the image dimensions
    # loop over the image, pixel by pixel
    for x in range(width):
        for y in range(height):
            r,g,b = img[x, y]
            counts[r] += 1

def count_pixels(folder_path):
    img_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    counts = [0]*13

    for img_path in img_paths:
        process_image(img_path, counts)
        print(counts)
    return counts


# path_rst = path/'results_test'
def cal_weights(lst):
    # calculate the sum of all the values in the list
    total = sum(lst)
    non_zero = [x for x in lst if x != 0]
    min_val = min(non_zero)
    # calculate the weight for each value
    for i in range(len(lst)):
        if lst[i] == 0:
            lst[i] = min_val

    weights = [(1/x)*(total) for x in lst]

    # divide all the weights by the weight of the first element
    weights = [w/weights[0] for w in weights]

    return weights

nums = count_pixels(path_rst)
print("---------------------")
print(cal_weights(nums))