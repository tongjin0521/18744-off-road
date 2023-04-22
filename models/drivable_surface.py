import os
import glob
import base64
import cv2 as cv
cv.__version__

from pathlib import Path
from tqdm import tqdm
import numpy as np
import timeit


drivable_res = 'test_res'
path = Path('../')
path_drst = path/drivable_res
path_drst.mkdir(exist_ok=True)
masks_folder = 'test_mask'
path_rst = path/masks_folder


def draw_drivable_region(frame):
    # Define colors
    white = (255, 255, 255)
    dark_green = (50, 205, 50)
    red = (0, 0, 255)
    grey = (128, 128, 128)
    # Create mask for driveable region and dangerous zones
    driveable_mask = np.logical_or.reduce((
                                            frame == (1, 1, 1), 
                                            frame == (2, 2, 2), 
                                            frame == (3, 3, 3)
                                          ))
    dangerous_mask = np.logical_or.reduce((frame == (5, 5, 5),
                                            frame == (11, 11, 11),
                                            frame == (12, 12, 12)))
    caution_mask = np.logical_or.reduce((  frame == (6, 6, 6), 
                                        frame == (7, 7, 7), 
                                        frame == (8, 8, 8), 
                                        frame == (9, 9, 9),
                                            frame == (10, 10, 10)))

    # If there is no roadmarking
    if not np.any(frame == (4, 4, 4)):
        # Create a new array filled with white
        result = np.full_like(frame, white)
        # Mark driveable region as dark green
        result[np.where(driveable_mask)[0], np.where(driveable_mask)[1], :] = dark_green
        # Mark dangerous zones as red
        result[np.where(dangerous_mask)[0], np.where(dangerous_mask)[1], :] = red

        result[np.where(caution_mask)[0], np.where(caution_mask)[1], :] = grey
    else:
        # Find the roadmarking
        roadmarking_mask = (frame == (4, 4, 4)).any(axis=2)
        # Find the right side of the road
        right_mask = np.zeros_like(roadmarking_mask, dtype=bool)
        for i in range(roadmarking_mask.shape[0]):
            right_mask[i, np.argmax(roadmarking_mask[i]):] = True
        # Apply driveable and dangerous masks to the right side of the road
        driveable_mask_right = np.logical_and(driveable_mask, right_mask[:, :, np.newaxis])
        dangerous_mask_right = np.logical_and(dangerous_mask, right_mask[:, :, np.newaxis])
        caution_mask_right = np.logical_and(caution_mask, right_mask[:, :, np.newaxis])
        # Create a new array filled with white
        result = np.full_like(frame, white)
        # Mark driveable region as dark green
        result[np.where(driveable_mask_right)[0], np.where(driveable_mask_right)[1], :] = dark_green
        # Mark dangerous zones as red
        result[np.where(dangerous_mask_right)[0], np.where(dangerous_mask_right)[1], :] = red
        result[np.where(caution_mask_right)[0], np.where(caution_mask_right)[1], :] = grey


    return result




# path_rst = path/'results_test'
filenames = [img for img in glob.glob(str(path_rst/"*.png"))]

filenames.sort() # ADD THIS LINE

for img in (filenames):
  frame = cv.imread(img)
  frame = draw_drivable_region(frame)
  name = f'{img}'.split("/")[-1]
  cv.imwrite(os.path.join(path_drst, name), frame)

print("Done!")