# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# 
# Modified by ddk
# --------------------------------------------------------

import action_datasets
from fast_rcnn.config_action import cfg
from utils.cython_bbox import bbox_overlaps

import PIL
import os, sys
import os.path as osp
import numpy as np
import scipy.sparse

class imdb(object):
  """Image database."""

  def __init__(self, name):
    # "rcnn_<datatype>_<imageset>"
    self._name = name
    # objects
    self._num_classes = 0
    self._classes = []
    # 
    self._image_index = []
    # 
    self._obj_proposer = 'selective_search'
    self._roidb = None
    self._roidb_handler = self.default_roidb
    # Use this dict for storing dataset specific config options
    self.config = {}


  # 
  # ######################################################################
  # 


  @property
  def name(self):
    return self._name

  @property
  def num_classes(self):
    return len(self._classes)

  @property
  def classes(self):
    return self._classes

  @property
  def image_index(self):
    return self._image_index

  @property
  def roidb_handler(self):
    return self._roidb_handler

  @roidb_handler.setter
  def roidb_handler(self, val):
    self._roidb_handler = val

  @property
  def roidb(self):
    # A roidb is a list of dictionaries, each with the following keys:
    #   boxes
    #   gt_overlaps
    #   gt_classes
    #   flipped
    if self._roidb is not None:
      return self._roidb
    # 
    self._roidb = self.roidb_handler()
    return self._roidb

  @property
  def cache_path(self):
    # path to the cache directory: data/cache/Actions/
    cache_path = osp.abspath(osp.join(action_datasets.ROOT_DIR, 'data', \
        'cache', 'Actions', self._datatype))
    if not os.path.exists(cache_path):
      os.makedirs(cache_path)
    # print cache_path
    # sys.exit(1)
    return cache_path

  @property
  def num_images(self):
    return len(self.image_index)

  def image_path_at(self, i):
    raise NotImplementedError

  def default_roidb(self):
    raise NotImplementedError


  # 
  # ######################################################################
  # 


  def append_flipped_images(self):
    # num_images = self.num_images
    num_images = 0
    if cfg.LESSEN_DEBUG_TIME:
      num_images = cfg.LESSEN_DEBUG_IMAGE_INDEX_LEN
    else:
      num_images = len(self._image_index)
    print "num_images:", num_images

    # Get images' width
    widths = [PIL.Image.open(self.image_path_at(i)).size[0]
              for i in xrange(num_images)]

    # Horizonly flipped
    for i in xrange(num_images):
      boxes = self.roidb[i]['boxes'].copy()
      oldx1 = boxes[:, 0].copy()
      oldx2 = boxes[:, 2].copy()
      boxes[:, 0] = widths[i] - oldx2 - 1
      boxes[:, 2] = widths[i] - oldx1 - 1
      # Check
      assert (boxes[:, 2] >= boxes[:, 0]).all()
      # Add filpped annotation info
      entry = {'boxes' : boxes,
               'gt_overlaps' : self.roidb[i]['gt_overlaps'],
               'gt_classes' : self.roidb[i]['gt_classes'],
               'flipped' : True}
      self.roidb.append(entry)
    # 
    self._image_index = self._image_index * 2


  # box_list: [[per_image_selective_boxes], ..., []]
  #   per_image_selective_boxes: [[x1 y1 x2 y2], ...., [x1 y1 x2 y2]] -> each proposal' box of one image
  # gt_roidb: {{'boxes' : boxes,'gt_classes': gt_classes, 'gt_overlaps' : overlaps, 'flipped' : False}}
  #   boxes: [[x1, y1, x2, y2], ..., [x1, y1, x2, y2]] -> shape: (obj_num, 4)
  #   gt_classes: [cls1 cls2, ...] -> shape: (obj_num), where clsi is a real value indicating the class type
  #   gt_overlaps: very sparse, shape: (obj_num, class_num), only gt_overlaps[on, cls] = 1, otherwise 0 for each row
  def create_roidb_from_box_list(self, box_list, gt_roidb):
    assert len(box_list) == self.num_images, \
            'Number of boxes must match number of ground-truth images'
    roidb = []
    for i in xrange(self.num_images):
      boxes = box_list[i]
      num_boxes = boxes.shape[0]      # the number of proposals
      overlaps = np.zeros((num_boxes, self.num_classes), dtype=np.float32)

      if gt_roidb is not None:
        gt_boxes = gt_roidb[i]['boxes']
        gt_classes = gt_roidb[i]['gt_classes']
        # (N * K)
        gt_overlaps = bbox_overlaps(boxes.astype(np.float),
                                    gt_boxes.astype(np.float))
        # return the index of gt class for each row
        argmaxes = gt_overlaps.argmax(axis=1)
        # return the value(1)
        maxes = gt_overlaps.max(axis=1)
        # 
        I = np.where(maxes > 0)[0]
      
        overlaps[I, gt_classes[argmaxes[I]]] = maxes[I]

      overlaps = scipy.sparse.csr_matrix(overlaps)
      roidb.append({'boxes' : boxes,
                    'gt_classes' : np.zeros((num_boxes,),
                                            dtype=np.int32),
                    'gt_overlaps' : overlaps,
                    'flipped' : False})
    return roidb


  @staticmethod
  def merge_roidbs(a, b):
    # a: ground truths
    # b: proposals
    assert len(a) == len(b)
    for i in xrange(len(a)):
      a[i]['boxes'] = np.vstack((a[i]['boxes'], b[i]['boxes']))
      a[i]['gt_classes'] = np.hstack((a[i]['gt_classes'],
                                      b[i]['gt_classes']))
      a[i]['gt_overlaps'] = scipy.sparse.vstack([a[i]['gt_overlaps'],
                                                 b[i]['gt_overlaps']])
    return a


  # 
  # ######################################################################
  # 


  def competition_mode(self, on):
    """Turn competition mode on or off."""
    pass