# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# 
# Modified by ddk
# --------------------------------------------------------

"""Pose Estimation Config System."""

import os, sys
import numpy as np
import os.path as osp
from easydict import EasyDict as edict


__GD = dict()
global_dict = __GD

__C = edict()
pose_cfg = __C

__C.TASK_NAME   = "Human Pose Estimation"
__C.RNG_SEED    = 3
__C.GPU_ID      = 0
__C.DEDUP_BOXES = 1./16.
__C.EPS         = 1e-14
__C.ROOT_DIR    = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))
__C.PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])


__C.TRAIN = edict()
__C.TRAIN.BATCH_SIZE = 1
__C.TRAIN.MIN_SCALE  = 240
__C.TRAIN.MAX_SCALE  = 256


__C.TEST = edict()
__C.TEST.DXY             = 0
__C.TEST.P_DXY           = 0    # person
__C.TEST.T_DXY           = 0    # torso
__C.TEST.TW_RATIO        = 0
__C.TEST.FILL_VAL        = 255
__C.TEST.BATCH_SIZE      = 1
__C.TEST.PARTS_NUM       = 7
__C.TEST.MIN_SIZE        = 240
__C.TEST.MAX_SIZE        = 256
__C.TEST.DRAW_TEXT       = False
__C.TEST.DISP_INFO       = False
__C.TEST.SHOW_IMGS       = False
__C.TEST.HAS_TORSO_MASK  = False
__C.TEST.DATA_LAYER_NAME       = ""
__C.TEST.TORSO_MASK_LAYER_NAME = ""
__C.TEST.AUX_INFO_LAYER_NAME   = ""
__C.TEST.TARGET_LAYER_NAME     = ""
__C.TEST.DRAW_PARTS_INDS = ""
__C.TEST.DRAW_SKELS_INDS = ""
__C.TEST.VIZ_DIRE        = ""
__C.TEST.PORT            = 8088
__C.TEST.ADDRESS         = "172.18.180.86"


__C.DISP_NUM     = 0
__C.SLEEP_TIME   = 2
__C.COMMA        = ","
__C.COLON        = ":"
__C.RADIUS       = 3
__C.R_THICKNESS  = 3
__C.L_THICKNESS  = 3
__C.VIZ_COLORS   = [
    (23,  119, 188), 
    (222,  12,  39), 
    (122, 212, 139), 
    (20,  198,  68), 
    (111,  12, 139), 
    (131, 112, 179), 
    (31,  211,  79), 
    (131, 121, 179), 
    (31,  121, 192), 
    (192,  21,  92), 
    (192,  21, 192), 
    (216, 121,  92), 
    (16,   11,  62), 
    (16,  111, 162), 
    (166,  71,  92), 
    (196, 111,  12), 
    (64,  181,  142), 
    (96,   46,  12), 
    (32,  224,  72),
    (132,  36, 112),
    (31,   11, 245),
    (49,  245,  16),
    (245,  21,  18),
]

__C.SOCKET = edict()
__C.SOCKET.BUFFER_SIZE = 2048
__C.SOCKET.SERVER_PORT = 8192
__C.SOCKET.SERVER_ADDR = "127.0.0.1"


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
                        'for config key: {}').format(type(b[k]), type(v), k))
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