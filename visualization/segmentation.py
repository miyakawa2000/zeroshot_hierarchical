import os
import glob
import cv2
import numpy as np
from utils.color import random_color

def visualize_separate_masks(bgr_img: np.ndarray, masks_dir: str, reverse=True) -> np.ndarray:
    # load masks
    mask_paths = glob.glob(os.path.join(masks_dir, "*.png"))        
    masks = []
    for mask_path in mask_paths:
        mask = cv2.imread(mask_path, -1)
        masks.append(mask)

    # sort by area
    masks = sorted(masks, key=lambda x: np.count_nonzero(x), reverse=reverse)
    
    # draw masks
    drawn = bgr_img.copy()
    for mask in masks:
        drawn[mask > 0] = random_color()
    
    return drawn