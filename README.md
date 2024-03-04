# D3T: Distinctive Dual-Domain Teacher Zigzagging Across RGB-Thermal Gap for Domain-Adaptive Object Detection
This is the PyTorch implementation of our paper: <br>
**D3T: Distinctive Dual-Domain Teacher Zigzagging Across RGB-Thermal Gap for Domain-Adaptive Object Detection**<br>
[Dinh Phat Do](https://github.com/EdwardDo69), [Taehoon Kim](https://scholar.google.com/citations?user=RrKoTX4AAAAJ&hl=en), [Jaemin Na](https://github.com/NaJaeMin92), Jiwon Kim, Keonho Lee, Kyunghwan Cho, [Wonjun Hwang](Wonjun Hwang)<br>
The IEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024<br>

[arXiv] [OpenReview] 

<p align="center">
<img src="./image/Figure2.jpg"><br>
</p>

# Installation

## Install PyTorch in Conda env

```
# Make environments via conda
conda create -n D3T python=3.8.5
conda activate D3T

#install pytorch
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
```
## Build cvpods

Please refers to [cvpods](https://github.com/Megvii-BaseDetection/cvpods).

```
# install cvpods
python3 -m pip install -e cvpods

# recommend wandb for visualizing the training
pip install wandb
pip install imgaug

# install numpy, setuptools
pip install numpy==1.20.3
pip install setuptools==59.5.0
```

# Dataset
All the data arrangements follow the format of PASCAL_VOC. The dataset files are in the folder of `cvpods/cvpods/data/datasets` and the config path are in `cvpods/cvpods/data/datasets/paths_route.py`. Please refers to [cvpods](https://github.com/Megvii-BaseDetection/cvpods).

**1. Download datasets**

**FLIR dataset**<br>
[[ Google Drive Link ]](https://drive.google.com/file/d/1xHDMGl6HJZwtarNWkEV3T4O9X4ZQYz2Y/view) This aligned dataset was firstly mentioned in Multispectral Fusion for Object Detection with Cyclic Fuse-and-Refine Blocks, ICIP 2020, Heng Zhang et al.<br>
We provide new ImageSets in `D3T/cvpods/data/FLIR_ICIP2020_aligned/ImageSets/Main/`
```
# Make VOC flir dataset from above dataset 
ln -s aligned/Annotations D3T/cvpods/data/FLIR_ICIP2020_aligned/
ln -s align/JPEGImages D3T/cvpods/data/FLIR_ICIP2020_aligned/
```

**KAIST dataset**<br>
[[ Project Link ]](https://soonminhwang.github.io/rgbt-ped-detection/) KAIST dataset: The KAIST Multispectral Pedestrian Dataset consists of 95k color-thermal pairs (640x480, 20Hz) taken from a vehicle.<br>
[[ Github Link ]](https://github.com/luzhang16/AR-CNN) This aligned anotations was firstly mentioned in Weakly Aligned Cross-Modal Learning for Multispectral Pedestrian Detection, ICCV 2019, Heng Zhang et al.<br>
We provide new ImageSets in `D3T/cvpods/data/kaist-KAIST_paired_VOC/ImageSets/Main/`

**2. Organize the dataset as following:**

```shell
D3T/
└── cvpods/
    └── data/
        ├── FLIR_ICIP2020_aligned/
        |    ├── Annotations/
        |    ├── ImageSets/
        |    |     └── Main/
        |    |         ├── align_train_CarPersonBicycle_IR.txt
        |    |         ├── align_train_CarPersonBicycle_RGB.txt
        |    |         └── align_validation.txt
        |    └── JPEGImages/
        └── KAIST_paired_VOC/
             ├── Annotations/
             ├── ImageSets/
             |     └── Main/
             |         ├── test_lwir.txt
             |         ├── trainval_RGB.txt
             |         └── trainval_Thermal.txt
             └── JPEGImages/

```

**3. Pretrained Model**<br>
We use the VGG16 and Res50 as the backbone, the pretrained model can be downloaded from this [[ Google Drive Link] ](https://drive.google.com/file/d/1Nb2sYh8GHiEUDtfUn5Buwugu6bNd1VbT/view?usp=sharing) and save its into `D3T/cvpods/pretrained_model/`

# Training
```
cd ./experiment/flir_rgb2thermal/

CUDA_VISIBLE_DEVICES=0,1,2,3 pods_train --dir . --dist-url "tcp://127.0.0.1:29007" --num-gpus 4 OUTPUT_DIR 'outputs/flir_rgb2thermal'
```
* The model is trained on 4 RTX A5000 GPUs.

# Testing
```
cd ./experiment/flir_rgb2thermal/

CUDA_VISIBLE_DEVICES=1 pods_test --num-gpus 1 --dir . --dist-url "tcp://127.0.0.1:29005" \
MODEL.WEIGHTS outputs/flir_rgb2thermal/best.pth \
OUTPUT_DIR outputs/flir_rgb2thermal.test
```

# Acknowledgement
This repo is developed based on Harmonious-Teacher, DenseTeacher and cvpods. Please check [Harmonious-Teacher](https://github.com/kinredon/Harmonious-Teacher), [DenseTeacher](https://github.com/Megvii-BaseDetection/DenseTeacher) and [cvpods](https://github.com/Megvii-BaseDetection/cvpods) for more details and features.

# Citation
If you think this work is helpful for your project, please give it a star and citation. We sincerely appreciate for your acknowledgments.
```

```
