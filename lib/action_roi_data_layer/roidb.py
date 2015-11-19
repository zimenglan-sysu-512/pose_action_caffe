# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# 
# Modified by ddk
# --------------------------------------------------------

"""Transform a roidb into a trainable roidb by adding a bunch of metadata."""

import numpy as np
from fast_rcnn.config_action import cfg
import utils.cython_bbox
import sys


# prepare_roidb
def prepare_roidb(imdb, phase=None):
  """
  Enrich the imdb's roidb by adding some derived quantities that
  are useful for training. This function precomputes the maximum
  overlap, taken over ground-truth boxes, between each ROI and
  each ground-truth box. The class with maximum overlap is also
  recorded.
  """
  roidb = imdb.roidb
  image_index_len = len(imdb.image_index)
  if cfg.LESSEN_DEBUG_TIME:
    if cfg.TRAIN.USE_FLIPPED and phase is not None and phase == cfg.TRAIN_PHASE:
      image_index_len = min(image_index_len, cfg.LESSEN_DEBUG_IMAGE_INDEX_LEN * 2)
    else:
      image_index_len = min(image_index_len, cfg.LESSEN_DEBUG_IMAGE_INDEX_LEN)
  print "image_index_len:", image_index_len

  # 
  for i in xrange(image_index_len):
    roidb[i]['image'] = imdb.image_path_at(i)
    # need gt_overlaps as a dense array for argmax
    gt_overlaps = roidb[i]['gt_overlaps'].toarray()
    # max overlap with gt over classes (columns)
    max_overlaps = gt_overlaps.max(axis=1)
    # gt class that had the max overlap (columns)
    max_classes = gt_overlaps.argmax(axis=1)
    # 
    roidb[i]['max_classes'] = max_classes
    roidb[i]['max_overlaps'] = max_overlaps
    # sanity checks
    # max overlap of 0 => class should be zero (background)
    zero_inds = np.where(max_overlaps == 0)[0]
    assert all(max_classes[zero_inds] == 0)
    # max overlap > 0 => class should not be zero (must be a fg class)
    nonzero_inds = np.where(max_overlaps > 0)[0]
    # print nonzero_inds.shape
    # print max_classes[nonzero_inds]
    assert all(max_classes[nonzero_inds] != 0)


# add_bbox_regression_targets
def add_bbox_regression_targets(roidb):
  """Add information needed to train bounding-box regressors."""
  assert len(roidb) > 0
  assert 'max_classes' in roidb[0], 'Did you call prepare_roidb first?'

  num_images = len(roidb)
  print "num_images:", num_images

  # Infer number of classes from the number of columns in gt_overlaps
  num_classes = roidb[0]['gt_overlaps'].shape[1]
  for im_i in xrange(num_images):
    rois = roidb[im_i]['boxes']
    max_overlaps = roidb[im_i]['max_overlaps']
    max_classes = roidb[im_i]['max_classes']
    # 
    roidb[im_i]['bbox_targets'] = \
            _compute_targets(rois, max_overlaps, max_classes)

  # Compute values needed for means and stds
  # var(x) = E(x^2) - E(x)^2
  class_counts = np.zeros((num_classes, 1)) + cfg.EPS
  sums = np.zeros((num_classes, 4))
  squared_sums = np.zeros((num_classes, 4))
  for im_i in xrange(num_images):
    targets = roidb[im_i]['bbox_targets']
    for cls in xrange(1, num_classes):
      cls_inds = np.where(targets[:, 0] == cls)[0]
      if cls_inds.size > 0:
        class_counts[cls] += cls_inds.size
        sums[cls, :] += targets[cls_inds, 1:].sum(axis=0)
        squared_sums[cls, :] += (targets[cls_inds, 1:] ** 2).sum(axis=0)

  means = sums / class_counts
  stds = np.sqrt(squared_sums / class_counts - means ** 2)

  # Normalize targets
  for im_i in xrange(num_images):
    targets = roidb[im_i]['bbox_targets']
    for cls in xrange(1, num_classes):
      cls_inds = np.where(targets[:, 0] == cls)[0]
      roidb[im_i]['bbox_targets'][cls_inds, 1:] -= means[cls, :]
      roidb[im_i]['bbox_targets'][cls_inds, 1:] /= stds[cls, :]

  # These values will be needed for making predictions
  # (the predicts will need to be unnormalized and uncentered)
  return means.ravel(), stds.ravel()



# rois: roidb[im_i]['boxes']
def _compute_targets(rois, overlaps, labels):
  """Compute bounding-box regression targets for an image."""
  # Ensure ROIs are floats
  rois = rois.astype(np.float, copy=False)

  # Indices of ground-truth ROIs
  gt_inds = np.where(overlaps == 1)[0]
  # 
  # Indices of examples for which we try to make predictions
  ex_inds = np.where(overlaps >= cfg.TRAIN.BBOX_THRESH)[0]

  # Get IoU overlap between each ex ROI and gt ROI
  # bbox_overlaps function return "(N, K) ndarray of overlap between boxes and query_boxes"
  #   where "boxes: (N, 4) ndarray of float, query_boxes: (K, 4) ndarray of float"
  ex_gt_overlaps = utils.cython_bbox.bbox_overlaps(rois[ex_inds, :], rois[gt_inds, :])
  
  # Find which gt ROI each ex ROI has max overlap with:
  # this will be the ex ROI's gt target
  gt_assignment = ex_gt_overlaps.argmax(axis=1)
  gt_rois = rois[gt_inds[gt_assignment], :]           # ???
  ex_rois = rois[ex_inds, :]
  ''' for example:
  rois.shape: (2448, 4)
  overlaps.shape: (2448,)
  labels.shape: (2448,)
  gt_inds.shape: (5,)
  ex_inds.shape: (51,)
  ex_gt_overlaps.shape: (51, 5)
  
  gt_assignment.shape: (51,)
  gt_rois.shape: (51, 4)
  ex_rois.shape: (51, 4)

  gt_inds: [0 1 2 3 4]
  ex_inds: [   0    1    2    3    4 1208 1229 1278 1280 1281 1282 1283 1299 1319 1320
   1321 1379 1381 1396 1397 1398 1413 1447 1456 1477 1478 1555 1556 1567 1745
   1747 1752 1756 1759 1760 1761 1762 1767 1779 1792 1844 1848 1849 1851 1852
   1857 1880 1881 1929 1983 2072]
  gt_assignment: [0 1 2 3 4 4 2 4 4 4 4 4 4 4 4 4 2 2 2 2 2 2 2 2 2 2 0 0 3 2 2 2 2 2 2 2 2
   2 2 2 1 1 1 1 1 2 1 1 2 1 1]
  gt_inds[gt_assignment]: [0 1 2 3 4 4 2 4 4 4 4 4 4 4 4 4 2 2 2 2 2 2 2 2 2 2 0 0 3 2 2 2 2 2 2 2 2
   2 2 2 1 1 1 1 1 2 1 1 2 1 1]

  '''

  ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + cfg.EPS
  ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + cfg.EPS
  ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
  ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

  gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + cfg.EPS
  gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + cfg.EPS
  gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
  gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights

  targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
  targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
  targets_dw = np.log(gt_widths / ex_widths)
  targets_dh = np.log(gt_heights / ex_heights)

  targets = np.zeros((rois.shape[0], 5), dtype=np.float32)
  targets[ex_inds, 0] = labels[ex_inds]
  targets[ex_inds, 1] = targets_dx
  targets[ex_inds, 2] = targets_dy
  targets[ex_inds, 3] = targets_dw
  targets[ex_inds, 4] = targets_dh
  return targets
