import os
import glob
import base64
import cv2 as cv
import cv2
cv.__version__

from pathlib import Path
from tqdm import tqdm
import numpy as np
import timeit


drivable_res = 'datasets/18102016_Part01_results_drivable'
path = Path('../')
path_drst = path/drivable_res
path_drst.mkdir(exist_ok=True)
masks_folder = 'datasets/18102016_Part01_results'
path_rst = path/masks_folder

def find_rightmost_pixel(ind,mask):
    # Find the lowest row containing the value of (4,4,4)
    lowest_row = -1
    for y in range(mask.shape[0]-1, -1, -1):
        if (4, 4, 4) in mask[y]:
            lowest_row = y
            break
    if lowest_row == -1:
        return None
    
    # Find the rightmost pixel in each row containing the value of (4,4,4)
    points = []
    for y in range(lowest_row, -1, -1):
        row = mask[y]
        if (4, 4, 4) in row:
            rightmost_x = np.max(np.where(row == (4, 4, 4))[0])
            points.append((rightmost_x, y))
    # Fit a curve to the points above the lowest row
    x_vals = [p[0] for p in points]
    y_vals = [p[1] for p in points]
    
    print(ind,len(x_vals))
    if (len(x_vals) < 40):
        return None
    coefficients = np.polyfit(y_vals, x_vals, deg=3)
    # Create a new array with the position of the fitted curve on each row
    new_mask = np.zeros(mask.shape[0])
    for y in range(mask.shape[0]-1, -1, -1):
        x_val = np.polyval(coefficients, y)
        new_mask[y] = max(0,int(x_val))
    return new_mask



def draw_drivable_region(ind,frame):
    # Define colors
    white = (255, 255, 255)
    dark_green = (50, 205, 50)
    light_green = (144, 238, 144)
    red = (0, 0, 255)
    light_red = (255, 102, 102)
    grey = (128, 128, 128)
    light_grey = (211, 211, 211)
    black = (0,0,0)
    # Create mask for driveable region and dangerous zones
    driveable_mask = np.logical_or.reduce((frame == (1, 1, 1), frame == (2, 2, 2), frame == (3, 3, 3)))
    dangerous_mask = np.logical_or.reduce((frame == (5, 5, 5),frame == (11, 11, 11),frame == (12, 12, 12),frame == (10, 10, 10)))
    caution_mask = np.logical_or.reduce((  frame == (6, 6, 6), frame == (7, 7, 7), frame == (8, 8, 8),frame == (9, 9, 9)))
    lane_mask = np.array((frame == (4, 4, 4)))
    curve_points = find_rightmost_pixel(ind,frame)
    # If there is no roadmarking
    if curve_points is None:
    # if True:
        # Create a new array filled with white
        result = np.full_like(frame, black)
        # Mark driveable region as dark green
        result[np.where(driveable_mask)[0], np.where(driveable_mask)[1], :] = dark_green
        # Mark dangerous zones as red
        result[np.where(dangerous_mask)[0], np.where(dangerous_mask)[1], :] = red
        # Mark other zones as grey
        result[np.where(caution_mask)[0], np.where(caution_mask)[1], :] = grey

        result[np.where(lane_mask)[0], np.where(lane_mask)[1], :] = white
    else:
        # Create a mask for the right side of the road
        right_mask = np.zeros_like(frame, dtype=bool)
        for i in range(right_mask.shape[0]):
            right_mask[i, int(curve_points[i]):] = True
        # Apply driveable and dangerous masks to the right side of the road
        driveable_mask_right = np.logical_and(driveable_mask, right_mask)
        dangerous_mask_right = np.logical_and(dangerous_mask, right_mask)
        caution_mask_right = np.logical_and(caution_mask, right_mask)
        # Create a new array filled with white
        result = np.full_like(frame, black)
        # Mark driveable region as dark green
        result[np.where(driveable_mask)[0], np.where(driveable_mask)[1], :] = light_green
        # Mark dangerous zones as red
        result[np.where(dangerous_mask)[0], np.where(dangerous_mask)[1], :] = light_red
        # Mark other zones as grey
        result[np.where(caution_mask)[0], np.where(caution_mask)[1], :] = light_grey

        result[np.where(lane_mask)[0], np.where(lane_mask)[1], :] = white
        # Mark driveable region as dark green
        result[np.where(driveable_mask_right)[0], np.where(driveable_mask_right)[1], :] = dark_green
        # Mark dangerous zones as red
        result[np.where(dangerous_mask_right)[0], np.where(dangerous_mask_right)[1], :] = red
        # Mark other zones as grey
        result[np.where(caution_mask_right)[0], np.where(caution_mask_right)[1], :] = grey


    return result




# path_rst = path/'results_test'
filenames = [img for img in glob.glob(str(path_rst/"*.png"))]

filenames.sort() # ADD THIS LINE

for ind,img in enumerate(filenames):
  frame = cv.imread(img)
  frame = draw_drivable_region(ind,frame)
  cv.imwrite(os.path.join(path_drst, os.path.basename(img)), frame)
print("Done!")