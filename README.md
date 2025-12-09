
<div align="center">

# D3T: Distinctive Dual-Domain Teacher Zigzagging Across RGB-Thermal Gap for Domain-Adaptive Object Detection CVPR2024
[![PWC](image/CVPR2024.jpg)](https://arxiv.org/abs/2403.09359)

[Dinh Phat Do](https://github.com/EdwardDo69), [Taehoon Kim](https://scholar.google.com/citations?user=RrKoTX4AAAAJ&hl=en), [Jaemin Na](https://github.com/NaJaeMin92), Jiwon Kim, Keonho Lee, Kyunghwan Cho, [Wonjun Hwang](https://scholar.google.co.uk/citations?user=-I8AfBAAAAAJ&hl=en)<br>

### [Paper](https://arxiv.org/abs/2403.09359) | [Youtube Oral](https://youtu.be/NPbjykByfRA?si=MbQUXWQ28rrT86AT)

> **Abstract:** *Domain adaptation for object detection typically entails transferring knowledge from one visible domain to another visible domain. However, there are limited studies on adapting from the visible to the thermal domain, because the domain gap between the visible and thermal domains is much larger than expected, and traditional domain adaptation can not successfully facilitate learning in this situation. To overcome this challenge, we propose a Distinctive Dual-Domain Teacher (D3T) framework that employs distinct training paradigms for each domain. Specifically, we segregate the source and target training sets for building dual-teachers and successively deploy exponential moving average to the student model to individual teachers of each domain. The framework further incorporates a zigzag learning method between dual teachers, facilitating a gradual transition from the visible to thermal domains during training. We validate the superiority of our method through newly designed experimental protocols with well-known thermal datasets, i.e., FLIR and KAIST.*

</div>

</br>


<p align="center">
<img src="/image/Figure2.jpg">
</p>

**Overview of D3T:** Our D3T model consists of two stages. **Burn-in Stage:** We initiate the training of the object detector using
labeled data from the RGB domain. **Zigzag Learning Stage:** Comprises two distinct and interleaved training components for the Thermal
domain and the RGB domain, respectively. During each step of training, the student model utilizes images from a single domain for
training but leverages knowledge from two teachers for enhanced learning effectiveness. In each step, only one teacher model is updated
corresponding to the trained domain.

</br>

<div align="center">
  
![Demo](/image/Demo.gif)

**Visualization of our D3T model and RGB source only model**
</div>

<br>
<br>

## Environments
```
# Prepare environments via conda
conda create -n D3T python=3.8.5
conda activate D3T
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
( OR 
conda install cudatoolkit=11.1 -c pytorch -c conda-forge
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
)

# install cvpods
git clone https://github.com/EdwardDo69/D3T.git
cd D3T
python3 -m pip install -e cvpods

# recommend wandb for visualizing the training
pip install wandb
pip install imgaug

# Install some spectial version
pip install numpy==1.20.3
pip install setuptools==59.5.0
pip install Pillow==9.2.0
pip install scikit-learn
```
## Dataset
All the data arrangements follow the format of PASCAL_VOC. The dataset files are in the folder of `cvpods/data/` and the config path are in `cvpods/cvpods/data/datasets/paths_route.py`. Please refers to [cvpods](https://github.com/Megvii-BaseDetection/cvpods).

#### Aligned Flir RGB -> Thermal

```sh
[data]
    ├── FLIR_ICIP2020_aligned
          ├── AnnotatedImages
          ├── Annotations
          ├── ImageSets
          └── JPEGImages
```

* Please download Aligned Flir dataset from the [link](https://drive.google.com/file/d/1xHDMGl6HJZwtarNWkEV3T4O9X4ZQYz2Y/view). Like above image, move `Annotations`, `JPEGImages` and `AnnotatedImages` to   `./cvpods/data`
* We make NEW ImageSets : cvpods/data/FLIR_ICIP2020_aligned/ImageSets

## Pretrained Model
We use the VGG16 as the backbone, the pretrained model can be downloaded from this [link](https://drive.google.com/file/d/1Nb2sYh8GHiEUDtfUn5Buwugu6bNd1VbT/view?usp=sharing). Then the `MODEL.WEIGHTS` should be updated in `config.py` correspondingly.

## Training
```
cd experiment/flir_rgb2thermal/
CUDA_VISIBLE_DEVICES=0,1,2,3 pods_train --dir . --dist-url "tcp://127.0.0.1:29007" --num-gpus 4 OUTPUT_DIR 'outputs/thermal'
```
* If you want use `wandb`, specify wandb account in `runner.py` and then add `WANDB True` into the command.
* The model is trained on 4 NVIDIA RTX A5000 GPUs.

## Testing
```
CUDA_VISIBLE_DEVICES=0 pods_test --dir . --num-gpus 1 MODEL.WEIGHTS $model_path
Ex:
CUDA_VISIBLE_DEVICES=1 pods_test --num-gpus 1 --dir . --dist-url "tcp://127.0.0.1:29055" \
MODEL.WEIGHTS D3T/experiment/flir_rgb2thermal/outputs/thermal/best.pth \
OUTPUT_DIR D3T/experiment/flir_rgb2thermal/outputs/test
```
Note that if you provide a relative model path, the `$model_path` is the relative path to `cvpods`. It is recommended to use the absolute path for loading the right model.

## Checkpoint
To facilitate the verification of our results, we provide our checkpoint for the `FLIR and KAIST dataset`. Please download it from the following [link](https://drive.google.com/drive/folders/1M2WoTiPSjuRRowM0IPqZbLETdD8EiM_l?usp=drive_link).

## Acknowledgement
This repo is developed based on Harmonious Teacher and cvpods. Please check [Harmonious Teacher](https://github.com/kinredon/Harmonious-Teacher) and [cvpods](https://github.com/Megvii-BaseDetection/cvpods) for more details and features.

## Citation
```
@inproceedings{do2024d3t,
  title={D3T: Distinctive Dual-Domain Teacher Zigzagging Across RGB-Thermal Gap for Domain-Adaptive Object Detection},
  author={Do, Dinh Phat and Kim, Taehoon and Na, Jaemin and Kim, Jiwon and Lee, Keonho and Cho, Kyunghwan and Hwang, Wonjun},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={23313--23322},
  year={2024}
}
```

## License
This repo is released under the Apache 2.0 license. Please see the LICENSE file for more information.

## Contact
For inquiries, please contact: phatai@ajou.ac.kr
