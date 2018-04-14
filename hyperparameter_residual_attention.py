# -*- coding: utf-8 -*-
"""
HyperParameter class
"""
from pathlib import Path
import os

class HyperParams:
    """Hyper-Parameters"""
    # data path
    HOME_DIR = Path(os.environ['HOME']) / 'residual-attention-network'
    DATASET_DIR = HOME_DIR / "dataset"
    DATASET_DIR.mkdir(exist_ok=True)

    name = 'cifar-10'
    output_dims = 10
    DATA_DIR = Path('/datadrive') / name
    IMAGE_DIR = DATA_DIR / 'original'
    SAVE_DIR = DATA_DIR / 'record'

    # dataset
    target_dataset = "CIFAR-10"

    # setting
    LR = 1e-1
    WEIGHT_DECAY = 5e-4
    RANDOM_STATE = 1234
    NUM_EPOCHS = 200
    BATCH_SIZE = 128
    VALID_BATCH_SIZE = 100
