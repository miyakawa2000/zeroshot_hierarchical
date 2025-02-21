# How to build

## 0. Build the container
```
bash docker_build.sh
bash docker_run.sh
```


## 1. Install CLIP
```
cd OVSeg/third_party/CLIP
python3 -m pip install -Ue .
cd ../../../
```
## 2. Install GroundingDINO
```
cd GroundingDINO
python3 -m pip install -e .
cd ../
```
## 3. Prepare the pre-trained weights of SAM, OVSeg, and Granouding DINO
```
mkdir ./weights/
wget -P ./weights/ https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
mkdir ./OVSeg/weights/
wget -P ./OVSeg/weights/ "https://drive.google.com/file/d/1cn-ohxgXDrDfkzC1QdO-fi8IjbjXmgKy/view?usp=sharing"
mkdir ./GroundingDINO/weights/
wget -P ./GroundingDINO/weights/ https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```

## in the case Error: "AttributeError: module 'PIL.Image' has no attribute 'LINEAR'"
```
python3 -m pip install -U 'git+https://github.com/facebookresearch/detectron2.git@ff53992b1985b63bd3262b5a36167098e3dada02'
```