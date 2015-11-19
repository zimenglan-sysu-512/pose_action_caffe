# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# 
# Modified by ddk
# --------------------------------------------------------

"""The data layer used during training to train a Fast R-CNN network.

RoIDataLayer implements a Caffe Python layer.
"""

import caffe
from fast_rcnn.config_action import cfg
from action_roi_data_layer.minibatch import get_minibatch
# 
import numpy as np
import yaml
from multiprocessing import Process, Queue



class RoIDataLayer(caffe.Layer):
  """Fast R-CNN data layer used for training."""


  # Called in ../fast_rcnn/train_action.py `__init__` function
  # That's to say, when bulid up the rcnn network, before training,
  # we must set roidb for RoIDataLayer
  def set_roidb(self, roidb):
    """Set the roidb to be used by this layer during training."""
    self._roidb = roidb
    self._shuffle_roidb_inds()

    # Use a prefetch thread in roi_data_layer.layer
    # So far I haven't found this useful; likely more engineering work is required
    if cfg.TRAIN.USE_PREFETCH:
      self._blob_queue = Queue(10)
      self._prefetch_process = BlobFetcher(self._blob_queue,
                                           self._roidb,
                                           self._num_classes)
      self._prefetch_process.start()
      # Terminate the child process when the parent exists
      def cleanup():
        print 'Terminating BlobFetcher'
        self._prefetch_process.terminate()
        self._prefetch_process.join()
      import atexit
      atexit.register(cleanup)


  # Setup
  def setup(self, bottom, top):
    """Setup the RoIDataLayer."""

    # parse the layer parameter string, which must be valid YAML
    layer_params = yaml.load(self.param_str_)

    self._num_classes = layer_params['num_classes']

    # Must place the ground truth / instance bounding box in the 1st position
    self._name_to_top_map = {'data': 0, 'rois': 1, 'labels': 2, 'n_rois_count': 3}

    self._test_n_ims = cfg.TRAIN.TEST_N_IMS
    self._has_visual_bbox = cfg.TRAIN.HAS_VISUAL_BBOX

    # top: 'data'
    # top: 'rois'
    # top: 'labels'
    # top: 'n_rois_count'
    # top: 'bbox_targets'
    # top: 'bbox_loss_weights'

    # shape[0] <==> num;
    # shape[1] <==> channels;
    # shape[2] <==> height;
    # shape[3] <==> width;
    # 
    # below `reshape` is mapping to ../../caffe_fast_rcnn/python/caffe/test/_caffe.cpp `Blob_Reshape`
    # the args in reshape will be the `bp::tuple args`

    # data blob: holds a batch of N images, each with 3 channels
    # The height and width (100 x 100) are dummy values
    self._first_ims_per_batch = cfg.TRAIN.IMS_PER_BATCH
    top[0].reshape(self._first_ims_per_batch, 3, 100, 100)  # means num = 1, channels = 3, height = 100, width = 100

    # rois blob: holds R regions of interest, each is a 5-tuple
    # (n, x1, y1, x2, y2) specifying an image batch index n and a
    # rectangle (x1, y1, x2, y2)
    self._batch_size = cfg.TRAIN.BATCH_SIZE
    top[1].reshape(self._batch_size, 5)    # means num = 1, channels = 5, height = 1, width = 1

    # labels blob: R categorical labels in [0, ..., K] for K foreground
    # classes plus background
    top[2].reshape(self._first_ims_per_batch)
    # 
    # record the number of secondary regions
    top[3].reshape(self._first_ims_per_batch)

    if cfg.TRAIN.BBOX_REG:
      self._name_to_top_map['bbox_targets'] = 4
      self._name_to_top_map['bbox_loss_weights'] = 5

      # bbox_targets blob: R bounding-box regression targets with 4
      # targets per class
      top[4].reshape(1, self._num_classes * 4)

      # bbox_loss_weights blob: At most 4 targets per roi are active;
      # thisbinary vector sepcifies the subset of active targets
      top[5].reshape(1, self._num_classes * 4)


  # 
  # #####################################################################
  # 


  def _shuffle_roidb_inds(self):
    """Randomly permute the training roidb."""
    if self._has_visual_bbox:
      print "test_n_ims:", self._test_n_ims
      self._perm = np.random.permutation(np.arange(self._test_n_ims))
    else:
      self._perm = np.random.permutation(np.arange(len(self._roidb)))
    self._cur = 0


  def _get_next_minibatch_inds(self):
    """Return the roidb indices for the next minibatch."""
    if self._has_visual_bbox:
      if self._cur + cfg.TRAIN.IMS_PER_BATCH >= self._test_n_ims:
        self._shuffle_roidb_inds()
    else:
      if self._cur + cfg.TRAIN.IMS_PER_BATCH >= len(self._roidb):
        self._shuffle_roidb_inds()

    db_inds = self._perm[self._cur: self._cur + cfg.TRAIN.IMS_PER_BATCH]
    self._cur += cfg.TRAIN.IMS_PER_BATCH
    return db_inds


  def _get_next_minibatch(self):
    """Return the blobs to be used for the next minibatch.

    If cfg.TRAIN.USE_PREFETCH is True, then blobs will be computed in a
    separate process and made available through self._blob_queue.
    """
    if cfg.TRAIN.USE_PREFETCH:
      return self._blob_queue.get()
    else:
      db_inds = self._get_next_minibatch_inds()
      minibatch_db = [self._roidb[i] for i in db_inds]
      # return get_minibatch(minibatch_db, self._num_classes)    # origin
      return get_minibatch(minibatch_db, self._num_classes, self._has_visual_bbox)


  # Forward
  def forward(self, bottom, top):
    """Get blobs and copy them into this layer's top blob vector."""
    blobs = self._get_next_minibatch()

    for blob_name, blob in blobs.iteritems():
      top_ind = self._name_to_top_map[blob_name]
      top[top_ind].reshape(*(blob.shape))
      top[top_ind].data[...] = blob.astype(np.float32, copy=False)
      # if blob_name == 'n_rois_count':
      #   print top[top_ind].data[...]


  # Backward
  def backward(self, top, propagate_down, bottom):
      """This layer does not propagate gradients."""
      pass

  # Reshape
  def reshape(self, bottom, top):
      """Reshaping happens during the call to forward."""
      pass



class BlobFetcher(Process):
  """Experimental class for prefetching blobs in a separate process."""
  def __init__(self, queue, roidb, num_classes):
    super(BlobFetcher, self).__init__()
    self._queue = queue
    self._roidb = roidb
    self._num_classes = num_classes
    self._perm = None
    self._cur = 0
    self._shuffle_roidb_inds()
    # fix the random seed for reproducibility
    np.random.seed(cfg.RNG_SEED)
    self._has_visual_bbox = False

  def _shuffle_roidb_inds(self):
    """Randomly permute the training roidb."""
    # TODO(rbg): remove duplicated code
    self._perm = np.random.permutation(np.arange(len(self._roidb)))
    self._cur = 0

  def _get_next_minibatch_inds(self):
    """Return the roidb indices for the next minibatch."""
    # TODO(rbg): remove duplicated code
    if self._cur + cfg.TRAIN.IMS_PER_BATCH >= len(self._roidb):
      self._shuffle_roidb_inds()

    db_inds = self._perm[self._cur:self._cur + cfg.TRAIN.IMS_PER_BATCH]
    self._cur += cfg.TRAIN.IMS_PER_BATCH
    return db_inds

  def run(self):
    print
    print 'BlobFetcher started'
    while True:
      db_inds = self._get_next_minibatch_inds()
      minibatch_db = [self._roidb[i] for i in db_inds]
      # 
      blobs = get_minibatch(minibatch_db, self._num_classes, self._has_visual_bbox)
      # 
      self._queue.put(blobs)
