import logging
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Literal

GENERATOR_CHANNELS_COUNT = 64
GENERATOR_GROWTHS_CHANNELS_COUNT = 32
GENERATOR_RES_DENSE_BLOCKS_COUNT = 3
GENERATOR_RRDB_COUNT = 23
GENERATOR_LARGE_KERNEL_SIZE = 9
GENERATOR_SMALL_KERNEL_SIZE = 3

DISCRIMINATOR_CHANNELS_COUNT = 64
DISCRIMINATOR_KERNEL_SIZE = 3
DISCRIMINATOR_CONV_BLOCKS_COUNT = 8
DISCRIMINATOR_LINEAR_LAYER_SIZE = 1024

SCALING_FACTOR: Literal[2, 4, 8] = 4
CROP_SIZE = 128
GRADIENT_ACCUMULATION_STEPS: Literal[1, 2, 4, 8] = 1
TRAIN_BATCH_SIZE = 16
REAL_TRAIN_BATCH_SIZE = TRAIN_BATCH_SIZE // GRADIENT_ACCUMULATION_STEPS
TEST_BATCH_SIZE = 1
GENERATOR_LEARNING_RATE = 1e-5
DISCRIMINATOR_LEARNING_RATE = 1e-5
EPOCHS = 500
PRINT_FREQUENCY = 200

SCHEDULER_MILESTONES = [150, 300]
SCHEDULER_SCALING_VALUE = 0.5

ADVERSARIAL_LOSS_SCALING_VALUE = 5e-3
PIXEL_LOSS_SCALING_VALUE = 1e-2

WEIGHTS_SCALING_VALUE = 0.1
RESIDUAL_SCALING_VALUE = 0.2

GRADIENT_CLIPPING_VALUE = 10.0
LEAKY_RELU_NEGATIVE_SLOPE_VALUE = 0.2

NUM_WORKERS = 8

TILE_SIZE = 512
TILE_OVERLAP = 64

LOAD_PSNR_CHECKPOINT = True
LOAD_BEST_PSNR_CHECKPOINT = False

INITIALIZE_WITH_PSNR_CHECKPOINT = True
LOAD_ESRGAN_CHECKPOINT = False
LOAD_BEST_ESRGAN_CHECKPOINT = False

DEV_MODE = False

TRAIN_DATASET_PATH = Path("data/DF2K_OST.txt")
VAL_DATASET_PATH = Path("data/DIV2K_valid.txt")
TEST_DATASETS_PATHS = [
    Path("data/Set5.txt"),
    Path("data/Set14.txt"),
    Path("data/BSDS100.txt"),
    Path("data/Urban100.txt"),
]

PSNR_CHECKPOINT_DIR_PATH = Path("checkpoints/psnr_latest")
BEST_PSNR_CHECKPOINT_DIR_PATH = Path("checkpoints/psnr_best")
ESRGAN_CHECKPOINT_DIR_PATH = Path("checkpoints/esrgan_latest")
BEST_ESRGAN_CHECKPOINT_DIR_PATH = Path("checkpoints/esrgan_best")

INFERENCE_INPUT_IMG_PATH = Path("images/inference_img_1.jpg")
INFERENCE_OUTPUT_IMG_PATH = Path("images/sr_img_1.png")
INFERECE_COMPARISON_IMG_PATH = Path("images/comparison_img_1.png")


def create_logger(
    log_level: str,
    caller_file_name: str,
    log_file_name: str | None = None,
    max_log_file_size: int = 5 * 1024 * 1024,
    backup_count: int = 10,
) -> logging.Logger:
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y.%m.%d %H:%M:%S"
    )

    logger.handlers.clear()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    logger.addHandler(console_handler)

    if not log_file_name:
        current_date = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        log_file_name = f"logs/srgan_{Path(caller_file_name).stem}_{current_date}.log"

    log_file_path = Path(log_file_name)
    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    file_handler = RotatingFileHandler(
        filename=log_file_path,
        maxBytes=max_log_file_size,
        backupCount=backup_count,
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
