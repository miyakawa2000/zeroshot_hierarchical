import os
import glob
import yaml
import argparse
import csv
import cv2
import numpy as np
from tqdm import tqdm

from utils.path import basename
from evaluation.metrics import area
from leafonlysam.LeafOnlySAM import istoobig, remove_toobig


def merge_masks_with_LSAM(masks_dir: str, min_area_th: float=0, max_area_th: float=np.inf, th_ratio=0.9, output_img_size =(1024, 1024)):
    """Merge leaf masks with NMS using SCORE

    Args:
        masks_dir (str): その画像のマスク (cv2.IMREAD_GRAYSCALEで読み込まれる) 全てを含むdir．ただしboundaryに乗るマスクは含まない．metadata.csvも含まれていること．
        th_IoU (float): Overlapを検知するIoUの閾値．
        th_area (int): 面積がth_area以下のマスクは無視する．
    """
    
    # load metadata
    metadata_filename = 'metadata.csv'
    csv_path = os.path.join(masks_dir, metadata_filename)
    with open(csv_path) as csvfile:
        reader = csv.DictReader(csvfile)
        metadata = [data for data in reader]
    metadata = sorted(metadata, key=lambda x: int(x['area']), reverse=True) # sort by area in descending order
    
    # load mask images
    masks = []
    for mask_data in metadata:
        # load mask
        try:
            mask_filename = mask_data['id'] + '.png'
        except:
            mask_filename = mask_data['filename']
        mask_path = os.path.join(masks_dir, mask_filename)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        masks.append(mask)
    
    # integration
    idx_toobig = istoobig(masks, th_ratio)
    masks_not_duplicate = remove_toobig(masks, idx_toobig, th_ratio)
    
    # make output
    output_img = np.zeros(output_img_size).astype(np.uint16) # init output img
    for i, mask in enumerate(masks_not_duplicate):
        if area(mask) > max_area_th or area(mask) < min_area_th:
            continue
        output_img[np.where(mask != 0)] = i+1
    
    return output_img

def make_leaf_seg(masks_dirs: list, output_dir: str, config, VALIDATION_MODE=False):
                  #th_ratio: float=0.7, 
                  #min_area_th: float=0, max_area_th: float=1024*1024, output_img_size =(1024, 1024), 
                  #VALIDATION_MODE: bool=False):
    """
    
    Args:
        masks_metrics (str, optional): 'stability_score' or 'predicted_iou' or 'area_negative'.


    """
    
    if not VALIDATION_MODE:
        os.makedirs(output_dir, exist_ok=True)
    else:
        outputs = []
    
    # load parameters
    with open(config) as f:
        config = yaml.safe_load(f)
        config = config['leaf_segmentation']
    min_area_th = config['min_area_th']
    max_area_th = config['max_area_th']
    th_ratio = config['th_ratio']
    output_img_size = config['output_img_size']
    
    # integration
    for masks_dir in tqdm(masks_dirs):
        leaf_segmentation_output = merge_masks_with_LSAM(masks_dir, min_area_th=min_area_th, max_area_th=max_area_th, th_ratio=th_ratio, output_img_size=output_img_size)
        img_name = basename(masks_dir) + '.png'
        # output
        if not VALIDATION_MODE:
            cv2.imwrite(os.path.join(output_dir, img_name), leaf_segmentation_output)
        else:
            outputs.append(leaf_segmentation_output)
    
    # output settings.txt
    if VALIDATION_MODE:
        return outputs
    else:
        # with open(os.path.join(output_dir, "settings.txt"), mode='w') as f:
        #     f.write("th_IoU: " + str(th_IoU) + "\n")
        #     f.write("masks_metrics: " + masks_metrics + "\n")
        #     f.write("min_area_th: " + str(min_area_th) + "\n")
        #     f.write("max_area_th: " + str(max_area_th) + "\n")
        return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--masks_dir", type=str, help="path to dir of leaf mask dirs")
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--config", type=str, help="path to config file (.yaml)")
    
    args = parser.parse_args()
    masks_dirs = glob.glob(os.path.join(args.masks_dir, "*"))
    output_dir = args.output_dir
    config_path = args.config
    
    make_leaf_seg(masks_dirs, output_dir, config_path)