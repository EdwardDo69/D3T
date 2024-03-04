from augmentations import StrongAug, WeakAug
from dataset import (FullCity, FullKITTI, FullSim10k, UnlabledCityCar,
                     UnlabledFoggy, UnlabledFlirAlignIr1to1, FlirAlignRgb)

from cvpods.configs.fcos_config import FCOSConfig

_config_dict = dict(
    DATASETS=dict(
        SUPERVISED=[
            (FlirAlignRgb, ),
        ],
        UNSUPERVISED=[
            (UnlabledFlirAlignIr1to1, ),
        ],
        TEST=("FlirAlignIr_val",),
    ),
    MODEL=dict(
        WEIGHTS='../../cvpods/pretrained_model/vgg16_wo_bn_caffe_normal.pkl',
        RESNETS=dict(DEPTH=50),
        BACKBONE=dict(NAME='vgg', VGG_W_BN=False, ),
        FPN=dict(IN_FEATURES=["vgg2", "vgg3", "vgg4"],),
        FCOS=dict(
            NUM_CLASSES=3,
            QUALITY_BRANCH='iou',
            CENTERNESS_ON_REG=False,
            NORM_REG_TARGETS=False,
            NMS_THRESH_TEST=0.6,
            BBOX_REG_WEIGHTS=(1.0, 1.0, 1.0, 1.0),
            FOCAL_LOSS_GAMMA=2.0,
            FOCAL_LOSS_ALPHA=0.25,
            IOU_LOSS_TYPE="iou",
            CENTER_SAMPLING_RADIUS=0,
            OBJECT_SIZES_OF_INTEREST=[
                [-1, 64],
                [64, 128],
                [128, 256],
                [256, 512],
                [512, float("inf")],
            ],
            NORM_SYNC=True,
            VFL=dict(
                USE_SIGMOID=True,
                ALPHA=0.75,
                GAMMA=2.0,
                WEIGHT_TYPE='iou',
                LOSS_WEIGHT=1.0
            ),
        ),
    ),
    DATALOADER=dict(
        NUM_WORKERS=4,
    ),
    SOLVER=dict(
        LR_SCHEDULER=dict(
            MAX_ITER=20000,
            STEPS=(70000, ),
            WARMUP_ITERS=1000,
            WARMUP_FACTOR=1.0 / 1000,
            GAMMA=0.1,
        ),
        OPTIMIZER=dict(
            BASE_LR=0.016,
        ),
        IMS_PER_BATCH=16,
        CHECKPOINT_PERIOD=5000,
        CLIP_GRADIENTS=dict(ENABLED=True)
    ),
    TRAINER=dict(
        NAME="SemiRunner",
        STEP2=10000,
        RGB_IR_CHANGE_TOTAL=200,
        RGB_IR_CHANGE_RGB=150,
        RGB_IR_CHANGE_INCREASE=50,
        EMA=dict(
            DECAY_FACTOR=0.9996,
            UPDATE_STEPS=1,
            START_STEPS=6000,  # adjust for 4-GPU setting
            FAKE=False
        ),
        SSL=dict(
            BURN_IN_STEPS=10000,  # adjust for 4-GPU setting
        ),
        DISTILL=dict(
            RATIO=0.01,
            HM=dict(
                ALPHA=0.5,
                BETA=0.5
            ),
            SUP_WEIGHT=1,
            UNSUP_WEIGHT=1,
            SUP_WEIGHT_STEP2=1,
            UNSUP_WEIGHT_STEP2=1,
            SUPPRESS='linear',
            WEIGHTS=dict(
                LOGITS=4.,
                DELTAS=1.,
                QUALITY=1.,
                VFL_UNSUP=1.,
                UHL=1.,
            ),
            UN_REGULAR_ALPHA=1.0,
            GAMMA=2.
        ),
        # WINDOW_SIZE=1,
    ),
    TEST=dict(
        EVAL_PERIOD=500,
    ),
    INPUT=dict(
        AUG=dict(
            TRAIN_PIPELINES=dict(
                SUPERVISED=(WeakAug, dict(short_edge_length=(640, 672, 704, 736, 768, 800), max_size=1333, sample_style="choice")),
                UNSUPERVISED=(StrongAug,)
            ),
            TEST_PIPELINES=[
                ("ResizeShortestEdge",
                 dict(short_edge_length=800, max_size=1333, sample_style="choice")),
            ],
        )
    ),
    OUTPUT_DIR='outputs',
    WANDB=False,
    GLOBAL=dict(
        LOG_INTERVAL=10,
    )
)


class CustomFCOSConfig(FCOSConfig):
    def __init__(self):
        super(CustomFCOSConfig, self).__init__()
        self._register_configuration(_config_dict)


config = CustomFCOSConfig()
