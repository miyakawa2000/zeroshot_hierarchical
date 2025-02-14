# Getting Started
## Installation
1. Clone this repository.
```
git clone https://github.com/miyakawa2000/zeroshot_hierarchical.git
```
2. Build the environment as `docker/README.md`.

# Run
## 1. Leaf Mask Collection
To collect leaf mask candidates, run `leaf_mask_collection.py` with
```
python3 leaf_mask_collection.py --dataset ${dataset_name} --mode ${val_or_test}
```
## 2. Leaf Instance Segmentation
To get leaf instance segmentation, run `leaf_segmentation.py` with
```
python3 leaf_segmentation.py --dataset ${dataset_name} --mode ${val_or_test}
```
## 3. Plant Instance Sementation
To get plant instance segmentation, run `plant_segmentation.py` with
```
python3 leaf_segmentation.py --dataset ${dataset_name} --mode ${val_or_test}
```
## 4. Evaluation
To evaluate the result, run hoge.py with
```
hoge
```
