import numpy as np


def area(mask: np.ndarray):
    return np.count_nonzero(mask)

def overlap(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    intersection_area = np.sum(intersection)
    return intersection_area

def maskIoU(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou = np.sum(intersection) / np.sum(union)
    return iou