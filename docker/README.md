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

## in the case Error: "AttributeError: module 'PIL.Image' has no attribute 'LINEAR'"
```
python3 -m pip install -U 'git+https://github.com/facebookresearch/detectron2.git@ff53992b1985b63bd3262b5a36167098e3dada02'
```