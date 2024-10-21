# How to build

1. Install requirements.txt
```
$ pip install -r requirements.txt
```

2. Clone OVSeg and rename the directory from 'ov-seg' to "OVSeg"
```
$ git clone https://github.com/facebookresearch/ov-seg.git
$ mv ov-seg OVSeg
```

3. Install CLIP
```
$ cd OVSeg/third_party/CLIP
$ pip install -Ue .
```

4. clone the GroundingDINO repository form GitHub
```
$ git clone https://github.com/IDEA-Research/GroundingDINO.git
```

5. Change the current directory to the GroundingDINO folder
```
$ cd GroundingDINO
```

6. Install the required dependencies in the current folder
```
$ pip install -e .
```

## in the case Error: "AttributeError: module 'PIL.Image' has no attribute 'LINEAR'"
```
$ python3 -m pip install -U 'git+https://github.com/facebookresearch/detectron2.git@ff53992b1985b63bd3262b5a36167098e3dada02'
```