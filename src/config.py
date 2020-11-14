import matplotlib
# Agg backend runs without a display
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
import sys
import json
import datetime
import numpy as np
#import skimage.io

#from sklearn.model_selection import train_test_split

# Import Mask RCNN


from src.utils import JSONConfig

# Load config
config_file = os.getenv('UNET_CONFIG', "default_config.json")
CONFIGS = JSONConfig(**json.load(open(config_file, 'r')))

# Dataset directory
DATA_DIR = CONFIGS.dataset_dir

# Number of classes
CLASSES = CONFIGS.classes

IMG_SIZE = CONFIGS.img_size

BATCH_SIZE = CONFIGS.batch_size
RESULT_DIR = CONFIGS.results_dir
LOG_DIR = CONFIGS.logs_dir