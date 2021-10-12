import os
import types
from pathlib import Path

config = types.SimpleNamespace()

# config
HOME_DIR = str(Path(__file__).parent)

# get home directory
config.HOME_DIR = HOME_DIR

# Subdirectory for downloading pretrained models
config.WEIGHTS = os.path.join(HOME_DIR, 'weights')

# Subdirectory name for saving trained weights
config.CKPTS_DIR = os.path.join(HOME_DIR, 'ckpts')

# Save log files during training
config.LOG_DIR = os.path.join(HOME_DIR, 'logs')

# Default path to the test image dataset files
config.DATASET_DIR = os.path.join(HOME_DIR, 'data')

# Default path to save result images
config.RESULT_DIR = os.path.join(HOME_DIR, 'results')

config.WEAPON_WARNING = [
    'Normal Things',
    'WARNING!! Pets Detected'
]
