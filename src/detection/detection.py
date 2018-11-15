import numpy as np
import math
import os
import cv2
import tensorflow as tf
import scipy.io
from object_detection.utils import label_map_util
from argparse import ArgumentParser

parser = ArgumentParser(description='Objection Detection.')

parser.add_argument('--video_root', required=True, help='Input video location.')