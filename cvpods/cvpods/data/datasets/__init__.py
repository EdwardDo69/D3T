#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2019-2021 Megvii Inc. All rights reserved.

from .citypersons import CityPersonsDataset
from .cityscapes import CityScapesDataset
from .coco import COCODataset
from .crowdhuman import CrowdHumanDataset
from .imagenet import ImageNetDataset
from .imagenetlt import ImageNetLTDataset
from .lvis import LVISDataset
from .objects365 import Objects365Dataset
from .torchvision_datasets import CIFAR10Dataset, STL10Datasets
from .voc import VOCDataset
from .widerface import WiderFaceDataset
from .city import CityDataset
from .foggy import FoggyDataset
from .city_car import CityCarDataset
from .sim10k import Sim10kDataset
from .kitti import KITTIDataset
from .city_7cls import City7clsDataset
from .bdd100k import Bdd100kDataset
from .bdd100k3class import Bdd100kDataset3Class
from .bdd100k3classBike import Bdd100kDataset3ClassBike
from .flir_align_ir import FlirAlignIr
from .flir_align_rgb import FlirAlignRgb
from .flir_full_ir import FlirFullIr
from .kaist_paired_rgb_voc import KaistPairedRgb
from .kaist_paired_thermal_voc import KaistPairedThermal
from .m3fd_ir import M3fdIr
from .m3fd_vis import M3fdVis
from .city_3class import CityDataset3Class
#from .city_car_faster import CityCarFasterDataset

__all__ = [k for k in globals().keys() if not k.startswith("_")]
