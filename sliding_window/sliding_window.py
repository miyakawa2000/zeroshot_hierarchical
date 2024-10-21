import os
from typing import Any, Dict, List
import cv2
import numpy as np
from utils.fileio import save_dict_list_as_csv

def sliding_window(img: np.ndarray, dsize: tuple) -> list:
    """imgをsliding windowにより分割した画像のリストを返す

    Args:
        img (np.ndarray): 分割する画像
        dsize (tuple): windowでcropした各画像をdsizeにresizeする

    Returns:
        crop_img_list (list): list of cropped imgs
    """
    img_h, img_w = img.shape[:2]
    win_w = int(img_w / 2)
    win_h = int(img_h / 2)
    x_step = int(win_w / 2)
    y_step = int(win_h / 2)
    crop_img_list = []
    # 以下任意のwin sizeおよびストライドに対応するには要改善
    for i in range(3):
        win_y = i * y_step
        for j in range(3):
            win_x = j * x_step
            crop_img = img[win_y:win_y+win_h, win_x:win_x+win_w]
            crop_img = cv2.resize(crop_img, dsize, interpolation=cv2.INTER_CUBIC)
            crop_img_list.append(crop_img)
    return crop_img_list

def get_window_corners(window_size):
    x_step = int(window_size[1] / 2)
    y_step = int(window_size[0] / 2)
    corners = []
    # 以下任意のwin sizeおよびストライドに対応するには要改善
    for i in range(3):
        win_y = i * y_step
        for j in range(3):
            win_x = j * x_step
            corners.append([win_y, win_x])
    return corners

def boundary(win_id):
    """
    return the boundary of window i.
    the number of window is 9.
    """
    if win_id == 0:
        return [1, 2]
    elif win_id == 1:
        return [1, 2, 3]
    elif win_id == 2:
        return [2, 3]
    elif win_id == 3:
        return [0, 1, 2]
    elif win_id == 4:
        return [0, 1, 2, 3]
    elif win_id == 5:
        return [0, 2, 3]
    elif win_id == 6:
        return [0, 1]
    elif win_id == 7:
        return [0, 1, 3]
    elif win_id == 8:
        return [0, 3]

def on_boundary(bbox, img_size, win_id):
    boundary_list = boundary(win_id)
    [x0, y0, w, h] = bbox
    H, W = img_size[0] - 1, img_size[1] - 1
    flag = False
    
    if 0 in boundary_list and y0 == 0:
        flag = True
    if 1 in boundary_list and x0 + w >= W:
        flag = True
    if 2 in boundary_list and y0 + h >= H:
        flag = True
    if 3 in boundary_list and x0 == 0:
        flag = True
    
    return flag

def covered_ids(merged_img, mask):
    """
    merged_img中のmaskの領域内の画素値(id)のリストを得る
    """
    covered_ids = merged_img[mask != 0].tolist()
    covered_ids = list(set(covered_ids)) # make it unique
    # remove 0 (bg)
    if 0 in covered_ids: 
        covered_ids.remove(0)
    return covered_ids

def overlap(mask1, mask2):
    """
    mask1とmask2の重なる領域の面積を求める
    """
    intersection = np.logical_and(mask1, mask2)
    intersection_area = np.sum(intersection)
    return intersection_area

def area(mask):
    return np.count_nonzero(mask)

def write_masks_to_folder(masks: List[Dict[str, Any]], path: str) -> None:
    header = "id,area,bbox_x0,bbox_y0,bbox_w,bbox_h,point_input_x,point_input_y,predicted_iou,stability_score,crop_box_x0,crop_box_y0,crop_box_w,crop_box_h"  # noqa
    metadata = [header]
    for i, mask_data in enumerate(masks):
        mask = mask_data["segmentation"]
        filename = f"{i}.png"
        cv2.imwrite(os.path.join(path, filename), mask * 255)
        mask_metadata = [
            str(i),
            str(mask_data["area"]),
            *[str(x) for x in mask_data["bbox"]],
            *[str(x) for x in mask_data["point_coords"][0]],
            str(mask_data["predicted_iou"]),
            str(mask_data["stability_score"]),
            *[str(x) for x in mask_data["crop_box"]],
        ]
        row = ",".join(mask_metadata)
        metadata.append(row)
    metadata_path = os.path.join(path, "metadata.csv")
    with open(metadata_path, "w") as f:
        f.write("\n".join(metadata))

    return

def save_masks_not_on_boundary(leaf_seg: list, save_folder: str, win_size: tuple=(512,512), min_area: int=0) -> None:
    """
    Args:
        leaf_seg (list): samの出力（全windowでのmask）
        save_folder (str): maskを保存するフォルダのdir
        win_size (tuple(int, int)): sliding windowのもとのsize．
    """    
    dsize = (1024, 1024) # size of output mask img (resize to dsize)
    corners = get_window_corners(win_size)
    
    metadata = []
    for i, mask_data in enumerate(leaf_seg):
        bbox = mask_data['bbox']
        win_id = mask_data['win_id']
        
        if win_id == 999: # not cropped
            mask_img = mask_data['segmentation']
            save_filename = str(i) + '.png'
            mask_area = area(mask_img)
            if mask_area < min_area:
                continue
            metadata.append({'filename': save_filename, 'area': mask_area, 'predicted_iou': mask_data['predicted_iou'], 'stability_score': mask_data['stability_score']})
            
            save_dir = os.path.join(save_folder, save_filename)
            cv2.imwrite(save_dir, mask_img * 255)
            
        else: # cropped
            if not on_boundary(bbox, dsize, win_id):
                mask_img = mask_data['segmentation']
                cv2.imwrite("../output/mask1.png", mask_img * 255) # save the mask temporary
                
                # resize the mask and move to appropriate loc
                mask_img = cv2.imread("../output/mask1.png", cv2.IMREAD_GRAYSCALE)
                mask_img = cv2.resize(mask_img, win_size, interpolation=cv2.INTER_CUBIC)
                resized_mask_img = np.zeros(dsize)
                win_y, win_x = corners[win_id]
                resized_mask_img[win_y:win_y+win_size[1], win_x:win_x+win_size[0]] = mask_img
                
                save_filename = str(i) + '.png'
                mask_area = area(resized_mask_img)
                if mask_area < min_area:
                    continue
                metadata.append({'filename': save_filename, 'area': mask_area, 'predicted_iou': mask_data['predicted_iou'], 'stability_score': mask_data['stability_score']})
                
                save_dir = os.path.join(save_folder, save_filename)
                cv2.imwrite(save_dir, resized_mask_img)
    
    # save metadata.csv
    if len(metadata) > 0:
        save_dict_list_as_csv(metadata, save_folder, 'metadata.csv')
    return