# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# 
# Modified by ddk
# --------------------------------------------------------

"""Fast R-CNN config system.

This file specifies default config options for Fast R-CNN. You should not
change values in this file. Instead, you should write a config file (in yaml)
and use cfg_from_file(yaml_file) to load it and override the default options.

Most tools in $ROOT/tools take a --cfg option to specify an override file.
  - See tools/{train,test}_net.py for example code that uses cfg_from_file()
  - See experiments/cfgs/*.yml for example YAML config override files
"""

import os
import os.path as osp
import numpy as np
# `pip install easydict` if you don't have it
from easydict import EasyDict as edict


# global dictionary to record some global variables for some convenience
# but restrict the keys must be difined in cfg
__GD = dict()
global_dict = __GD



__C = edict()
# Consumers can get config by:
#   from fast_rcnn_config import cfg
cfg = __C
# 
__C.TRAIN_BBOX_MEANS = "bbox_means"
__C.TRAIN_BBOX_STDS = "bbox_stds"
__C.TEST_BBOX_MEANS = "bbox_means"
__C.TEST_BBOX_STDS = "bbox_stds"
# 
__C.CLASS_TO_IND = "class_to_ind"
__C.IND_TO_CLASS = "ind_to_class"
__C.NUM_CLASSES = "num_classes"
__C.EXPERIMENT_NAME = "exper_name"
# 
__C.TRAIN_NUM_IMAGES = "num_images"
__C.TEST_NUM_IMAGES = "num_images"

__C.TRAIN_IMDB = "train_imdb"
__C.TRAIN_ROIDB = "train_roidb"
__C.TEST_IMDB = "test_imdb"
__C.TEST_ROIDB = "test_roidb"

__C.TRAIN_OUTPUT_DIR = "train_output_dir"
__C.TEST_OUTPUT_DIR =  "test_output_dir"


__C.TRAIN_IMDB_NAME = "train_imdb_name"
__C.TEST_IMDB_NAME = "test_imdb_name"


__C.LESSEN_DEBUG_TIME = False
__C.LESSEN_DEBUG_STR = "lessen_roidb"
__C.LESSEN_DEBUG_IMAGE_INDEX_LEN = 20


__C.TRAIN_PHASE = "TRAIN"
__C.TEST_PHASE = "TEST"
# 
# #####################################################################
# 


#
# Training options
#

__C.TRAIN = edict()

# Scales to use during training (can list multiple scales)
# Each scale is the pixel size of an image's shortest side
__C.TRAIN.SCALES = (600,)

# Max pixel size of the longest side of a scaled input image
__C.TRAIN.MAX_SIZE = 1000



# #########################################################
__C.TRAIN.HAS_SECONDARY_REGIONS = True
__C.TRAIN.NEED_WHOLE_IMAGE_BBOX = False
__C.TRAIN.IMS_PER_BATCH = 2  # Images to use per minibatch
__C.TRAIN.BATCH_SIZE = 30   # Minibatch size (number of regions of interest [ROIs])
__C.TRAIN.MIN_BATCH_SIZE_PER_IMAGE = \
    __C.TRAIN.BATCH_SIZE / __C.TRAIN.IMS_PER_BATCH

# Fraction of minibatch that is labeled foreground (i.e. class > 0)
# please see the paper "Contextual Action Recognition with R*CNN"
__C.TRAIN.FG_NUMBER = 14
__C.TRAIN.FG_FRACTION = 0.667   # 10 / 15

# Overlap threshold for a ROI to be considered a `secondary region`
# if >= FG_THRESH or if >= FG_THRESH_LO && if <= FG_THRESH_HI
__C.TRAIN.FG_THRESH = 0.5
__C.TRAIN.FG_THRESH_LO = 0.5
__C.TRAIN.FG_THRESH_HI = 0.85
__C.TRAIN.FG_ITSELF_THRESH = 1.0   # Primary region threshold

# Overlap threshold for a ROI to be considered background (
# class = 0 if overlap in [LO, HI)
__C.TRAIN.BG_THRESH_LO = 0.1
__C.TRAIN.BG_THRESH_HI = 0.5

# Use horizontally-flipped images during training?
__C.TRAIN.USE_FLIPPED = True

# Train bounding-box regressors
__C.TRAIN.BBOX_REG = False

# Overlap required between a ROI and ground-truth box in order for that ROI to
# be used as a bounding-box regression training example
__C.TRAIN.BBOX_THRESH = 0.5

# Iterations between snapshots
__C.TRAIN.SNAPSHOT_ITERS = 4000

# solver.prototxt specifies the snapshot path prefix, this adds an optional
# infix to yield the path: <prefix>[_<infix>]_iters_XYZ.caffemodel
__C.TRAIN.SNAPSHOT_INFIX = ''

# Use a prefetch thread in roi_data_layer.layer
# So far I haven't found this useful; likely more engineering work is required
__C.TRAIN.USE_PREFETCH = False

# 
cfg.TRAIN.TEST_N_IMS = 7
__C.TRAIN.HAS_VISUAL_BBOX = False
# Used for debug when `HAS_VISUAL_BBOX` is set true and if > 0, otherwise do nothing
__C.TRAIN.VISUAL_BBOX_LEN = 0

# 
# #####################################################################
# 


__C.TEST = edict()

# The number of iterations between two testing phases.
__C.TEST.TEST_INTERVAL = 10000
__C.TEST.START_TEST_ITER = 10000

# Images to use per minibatch
__C.TEST.IMS_PER_BATCH = 1

# Minibatch size (number of regions of interest [ROIs])
# if set to be -1, then use all secondary regions
__C.TEST.BATCH_SIZE = 15
__C.TEST.MIN_BATCH_SIZE_PER_IMAGE = __C.TEST.BATCH_SIZE / __C.TEST.IMS_PER_BATCH

# Fraction of minibatch that is labeled foreground (i.e. class > 0)
# Primary region: 1
# Secondary region N: 10
# BG region: 4
# 30 = 2 * (1 + 10 + 4)
# please see the paper "Contextual Action Recognition with R*CNN"
__C.TEST.FG_NUMBER = 10
__C.TEST.FG_FRACTION = 0.667   # 10 / 15


# Overlap threshold for a ROI to be considered a `secondary region`
# if >= FG_THRESH or if >= FG_THRESH_LO && if <= FG_THRESH_HI
__C.TEST.FG_THRESH = 0.5
__C.TEST.FG_THRESH_HI = 0.75
__C.TEST.FG_THRESH_LO = 0.5
__C.TEST.FG_ITSELF_THRESH = 1.0   # Primary region threshold


# Overlap threshold for a ROI to be considered background (
# class = 0 if overlap in [LO, HI)
__C.TEST.BG_NUMBER = 4
__C.TEST.BG_THRESH_HI = 0.1
__C.TEST.BG_THRESH_LO = 0.5


# Primary region threshold
__C.TEST.SR_ITSELF_THRESH = 1.0

# Used to control whether use secondary regions, if not, 
# Please be cafeful about your solver/train/test/ prototxt
__C.TEST.HAS_SECONDARY_REGIONS = True

# Threshold values to get the foreground secondary regions
__C.TEST.HAS_FG_SECONDARY_REGIONS = True
__C.TEST.SR_FG_THRESH_LO = 0.2
__C.TEST.SR_FG_THRESH_HI = 0.75

# Threshold values to get the background secondary regions
__C.TEST.HAS_BG_SECONDARY_REGIONS = False
__C.TEST.SR_BG_THRESH_LO = 0.1
__C.TEST.SR_BG_THRESH_HI = 0.45

# Can be the one of the secondary reions
__C.TEST.NEED_WHOLE_IMAGE_BBOX = False


# when testing, the secondary regions' scales must be the 
# same as the primary region's scale or not
# the secondary regions must match the scale of the primary region
__C.TEST.HAS_CONTROL_THE_SAME_SCALE = False

# Since each instance can have many secondary regions, and limited by the gpu
# When test one instance, we iterately run the primary region and secondary regions
# This variable controls the number of regions of one iteration
__C.TEST.SR_PER_NUMBER = 30

# where fg_ing.size < 1, them copy the primary region to the secondary regions 
# use SR_FG_COPY_PER_NUMBER to control the amount
__C.TEST.SR_FG_COPY_PER_NUMBER = 1

# Used for testing the datalayer to visualize the bboxes
__C.TEST.TEST_N_IMS = 7

# Control whether show the results of the primary region and the max-selected secondary region
__C.TEST.HAS_SHOW_REGION = False
# If show the result, control whether show the secondary region
__C.TEST.HAS_SHOW_SECONDARY_REGION = False

# Used for debug when create the primary region or secondary regions
# wchich are then fed into the datalayer
__C.TEST.HAS_VISUAL_BBOX = False

# Used for debug when `HAS_VISUAL_BBOX` is set true and if > 0, otherwise do nothing
__C.TEST.VISUAL_BBOX_LEN = 0

# Scales to use during testing (can list multiple scales)
# Each scale is the pixel size of an image's shortest side
__C.TEST.SCALES = (600,)     # origin: 600

# Max pixel size of the longest side of a scaled input image
__C.TEST.MAX_SIZE = 1000

# Overlap threshold used for non-maximum suppression (suppress boxes with
# IoU >= this threshold)
__C.TEST.NMS = 0.3

# Experimental: treat the (K+1) units in the cls_score layer as linear
# predictors (trained, eg, with one-vs-rest SVMs).
__C.TEST.SVM = False

# Test using bounding-box regressors
__C.TEST.BBOX_REG = False


# 
__C.TEST.PRIMARY_SECONDARY_VISUAL="primary_secondary_visual"


# 
# #####################################################################
# 


# The mapping from image coordinates to feature map coordinates might cause
# some boxes that are distinct in image space to become identical in feature
# coordinates. If DEDUP_BOXES > 0, then DEDUP_BOXES is used as the scale factor
# for identifying duplicate boxes.
# 1/16 is correct for {Alex,Caffe}Net, VGG_CNN_M_1024, and VGG16
__C.DEDUP_BOXES = 1./16.

# Pixel mean values (BGR order) as a (1, 1, 3) array
# These are the values originally used for training VGG16
__C.PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])

# For reproducibility
__C.RNG_SEED = 3

# A small number that's used many times
__C.EPS = 1e-14

# Root directory of project
__C.ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))

# Place outputs under an experiments directory
__C.EXP_DIR = 'default'



# 
# #####################################################################
# 


def get_output_dir(imdb, sub_dir = "output", net = None):
  """
  Return the directory where experimental artifacts are placed.
  A canonical path is built using the name from an imdb and a network
  (if not None).
  """
  datatype = imdb.name
  datatype = datatype.split("_")[1]

  path = osp.abspath(osp.join(__C.ROOT_DIR, sub_dir, datatype, imdb.name))
  if net is None:
    return path
  else:
    return osp.join(path, net.name)


def get_output_dir2(imdbname, sub_dir = "output", net = None):
  """
  Return the directory where experimental artifacts are placed.
  A canonical path is built using the name from an imdb and a network
  (if not None).
  """
  datatype = imdbname
  datatype = datatype.split("_")[1]
  path = osp.abspath(osp.join(__C.ROOT_DIR, sub_dir, datatype, imdbname))
  if net is None:
    return path
  else:
    return osp.join(path, net.name)


def _merge_a_into_b(a, b):
  """Merge config dictionary a into config dictionary b, clobbering the
  options in b whenever they are also specified in a.
  """
  if type(a) is not edict:
    return

  for k, v in a.iteritems():
    # a must specify keys that are in b
    if not b.has_key(k):
      raise KeyError('{} is not a valid config key'.format(k))

    # the types must match, too
    if type(b[k]) is not type(v):
      raise ValueError(('Type mismatch ({} vs. {}) '
                          'for config key: {}').format(type(b[k]),
                                                       type(v), k))
    # recursively merge dicts
    if type(v) is edict:
      try:
        _merge_a_into_b(a[k], b[k])
      except:
        print('Error under config key: {}'.format(k))
        raise
    else:
      b[k] = v


def cfg_from_file(filename):
  """Load a config file and merge it into the default options."""
  import yaml
  with open(filename, 'r') as f:
    yaml_cfg = edict(yaml.load(f))

  _merge_a_into_b(yaml_cfg, __C)