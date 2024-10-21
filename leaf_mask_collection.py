import multiprocessing as mp

import os
import glob
import argparse
from tqdm import tqdm

import cv2
import numpy as np
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.data.detection_utils import read_image

from OVSeg.open_vocab_seg import add_ovseg_config
from OVSeg.open_vocab_seg.utils import SAMVisualizationDemo
from utils.path import filename_wo_ext
from sliding_window.sliding_window import sliding_window, save_masks_not_on_boundary

def setup_cfg(config_file):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_ovseg_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.freeze()
    return cfg

def inferance_with_slidingwindow(class_names: str, trg_class_names: str, img_paths: list, granularity: float, dsize: tuple, min_mask_area: int, output_dir: str):
    
    mp.set_start_method("spawn", force=True)
    config_file = './OVSeg/configs/ovseg_swinB_vitL_demo.yaml'
    cfg = setup_cfg(config_file)
    class_names = class_names.split(',')
    trg_class_names = trg_class_names.split(',')
    demo = SAMVisualizationDemo(cfg, granularity, './weights/sam_vit_h_4b8939.pth', './OVSeg/weights/ovseg_clip_l_9a1909.pth')
    
    for img_path in tqdm(img_paths):
        
        # load img and crop it with sliding window
        img = read_image(img_path, format="BGR")
        crop_imgs = sliding_window(img, dsize)
        
        # inference on raw img
        leaf_segs = []
        _, _, ins_segs = demo.run_on_image(img, class_names)        
        for cls_idx, ins_seg in enumerate(ins_segs):
            if class_names[cls_idx] in trg_class_names:
                for item in ins_seg:
                    item['win_id'] = 999 # id for not cropped (raw) img
                leaf_segs.extend(ins_seg)
        
        # inference on each cropped img
        for crop_id, crop_img in enumerate(crop_imgs):
            
            _, _, ins_segs = demo.run_on_image(crop_img, class_names)
            
            # extend leaf_segs with masks of each cropped img
            for cls_idx, ins_seg in enumerate(ins_segs):
                if class_names[cls_idx] in trg_class_names:
                    for item in ins_seg:
                        item['win_id'] = crop_id
                    leaf_segs.extend(ins_seg)
        
        # save masks not on boundary
        img_name = filename_wo_ext(img_path)
        if len(trg_class_names) == 1:
            save_folder = os.path.join(output_dir, img_name)
            os.makedirs(save_folder, exist_ok=True)
            save_masks_not_on_boundary(leaf_segs, save_folder, win_size=(512,512), min_area=min_mask_area)
        else:
            for cls_idx, trg_class_name in enumerate(trg_class_names):
                save_folder = os.path.join(output_dir, img_name, trg_class_name)
                os.makedirs(save_folder, exist_ok=True)
                save_masks_not_on_boundary(leaf_segs[cls_idx], save_folder, win_size=(512,512), min_area=min_mask_area)
    
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    
    parser.add_argument("--class_names", type=str, default="green leaf,soil")
    parser.add_argument("--trg_class_names", type=str, default="green leaf")
    parser.add_argument("--gran", type=float, default=0.8)
    parser.add_argument("--min_mask_area", type=int, default=0)
    parser.add_argument("--dsize", type=tuple, default=(1024, 1024), help="Each cropped img will be resized to dsize.")
    
    args = parser.parse_args()
    
    class_names = args.class_names
    trg_class_names = args.trg_class_names
    granularity = args.gran
    min_mask_area = args.min_mask_area
    dsize = args.dsize
    
    img_paths = glob.glob(os.path.join(args.img_dir, "*.png"))
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    inferance_with_slidingwindow(class_names, trg_class_names, img_paths, granularity, dsize, min_mask_area, output_dir)
    