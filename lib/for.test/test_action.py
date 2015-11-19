# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Test a Fast R-CNN network on an imdb.roidb (image database)."""

import caffe

# for action recognition task in still/static images
from fast_rcnn.config_action import cfg, get_output_dir
from utils.cython_nms import nms
from utils.blob import im_list_to_blob

import argparse
from utils.timer import Timer
import numpy as np
from numpy import random as npr
import cv2
import cPickle
import heapq
import os, sys
import matplotlib.pyplot as plt



# 
# input: one image
# output: image pyramid and corresponding scale factors
def _get_image_blob(im):
  """Converts an image into a network input.

  Arguments:
    im (ndarray): a color image in BGR order

  Returns:
    blob (ndarray): a data blob holding an image pyramid
    im_scales (list): list of image scales (relative to im) used
      in the image pyramid
  """
  im_orig = im.astype(np.float32, copy=True)
  im_orig -= cfg.PIXEL_MEANS

  im_shape = im_orig.shape
  im_size_min = np.min(im_shape[0:2])
  im_size_max = np.max(im_shape[0:2])

  processed_ims = []
  im_scales = []

  for target_size in cfg.TEST.SCALES:
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
      im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
    # Rescale
    im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                    interpolation=cv2.INTER_LINEAR)
    # Rescaled factor
    im_scales.append(im_scale)
    # Get rescaled image
    processed_ims.append(im)

  # Create a blob to hold the input images
  # Here treat the pyramid of one image as a batch??
  blob = im_list_to_blob(processed_ims)

  return blob, np.array(im_scales)


# 
# get the scale close to 224 * 224 by using the diff_area between them
# input: original positions of bbox of roi (x1, y1, x2, y2), and scales of pyramid
# output: the rescaled positions of bbox or roi and the corresponding scaled index
def _project_im_rois(im_rois, scales):
  """Project image RoIs into the image pyramid built by _get_image_blob.

  Arguments:
      im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
      scales (list): scale factors as returned by _get_image_blob

  Returns:
      rois (ndarray): R x 4 matrix of projected RoI coordinates
      levels (list): image pyramid levels used by each projected RoI
  """
  im_rois = im_rois.astype(np.float, copy=False)

  if len(scales) > 1:
    widths = im_rois[:, 2] - im_rois[:, 0] + 1
    heights = im_rois[:, 3] - im_rois[:, 1] + 1

    areas = widths * heights
    scaled_areas = areas[:, np.newaxis] * (scales[np.newaxis, :] ** 2)
    diff_areas = np.abs(scaled_areas - 224 * 224)
    # print "diff_areas:", diff_areas
    levels = diff_areas.argmin(axis=1)[:, np.newaxis]
    # print "np.newaxis:", np.newaxis
    # print "scales.shape:", scales.shape
    # print "diff_areas:", diff_areas
    # print "levels:", levels
    # sys.exit(1)
  else:
    levels = np.zeros((im_rois.shape[0], 1), dtype=np.int)
    # print "im_rois.shape:", im_rois.shape
    # print "levels.shape:", levels.shape
    # sys.exit(1)

  # scale
  rois = im_rois * scales[levels]

  return rois, levels


# 
# input:
#   (x1, y1, x2, y2)
#   (scale_factor)
# output:
#   (scale_idx, sx1, sy1, sx2, sy2)
def _get_rois_blob(im_rois, im_scales):
  """Converts RoIs into network inputs.

  Arguments:
    im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
    im_scales (list): scale factors as returned by _get_image_blob

  Returns:
    blob (ndarray): R x 5 matrix of RoIs in the image pyramid
  """
  rois, levels = _project_im_rois(im_rois, im_scales)
  rois_blob = np.hstack((levels, rois))
  return rois_blob.astype(np.float32, copy=False)


# 
# input
#   bbox_target_data: contain the origin proposals clss and bbox positions
# return 
#   i.e. only one class has non-zero targets, 
#   that is to say, each of those fg proposals has bbox positions
#   the rest is reset to zero
def _get_bbox_regression_labels(bbox_target_data, num_classes):
  """
  Bounding-box regression targets are stored in a compact form in the
  roidb.

  This function expands those targets into the 4-of-4*K representation used
  by the network (i.e. only one class has non-zero targets). The loss weights
  are similarly expanded.

  Returns:
    bbox_target_data (ndarray):  N x 4K blob of regression targets
    bbox_loss_weights (ndarray): N x 4K blob of loss weights
  """
  clss = bbox_target_data[:, 0]
  bbox_targets = np.zeros((clss.size, 4 * num_classes), dtype=np.float32)
  bbox_loss_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
  inds = np.where(clss > 0)[0]
  for ind in inds:
    cls = clss[ind]
    start = 4 * cls
    end = start + 4
    bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
    bbox_loss_weights[ind, start:end] = [1., 1., 1., 1.]
  return bbox_targets, bbox_loss_weights


# 
# here, number of minibatch must be 1
def _get_minibatch(roidb, num_classes, has_visual_bbox = False):
  """Given a roidb, construct a minibatch sampled from it."""
  # 
  num_images = len(roidb)
  assert(num_images, 1)
  roidb = roidb[0]

  # 
  # Get image pyramid and corresponding scales and 
  # indices of scales(recording the suited scale 224 * 224)
  im = cv2.imread(roidb['image'])
  boxes = roidb['boxes']
  im_blobs, im_scales = _get_image_blob(im)

  # 
  # Sample primary region and the secordary regions
  # label = class RoI has max overlap with
  labels = roidb['max_classes']
  overlaps = roidb['max_overlaps']
  rois = roidb['boxes']
  rois = _get_rois_blob(rois, im_scales)
  
  # 
  # Select the primary region
  n_primary_region = 1
  fg_itself_inds = np.where(overlaps == cfg.TEST.FG_ITSELF_THRESH)[0]
  if fg_itself_inds.size > n_primary_region:
    # Will keep the same label ?? -- i worry about it
    fg_itself_inds = npr.choice(fg_itself_inds, size=n_primary_region,
                         replace=False)
  if fg_itself_inds.size != n_primary_region \
      or overlaps[fg_itself_inds[0]] != cfg.TEST.FG_ITSELF_THRESH:
    print "#########################################"
    print "fg_itself_inds.size:", fg_itself_inds.size
    print "fg_itself_inds:", fg_itself_inds[0]
    print
    print "overlaps:", overlaps
    print
    print "rois:", rois
    print
    print "missing primary regoin in \
        ${fast-rcnn-acton}/lib/action_roi_data_layer/minibatch \
        in `_sample_rois` function"
    sys.exit(1)

  # 
  # Select the secondary region sets as foreground
  fg_inds = np.where((overlaps >= cfg.TEST.FG_THRESH_LO) & 
          (overlaps <= cfg.TEST.FG_THRESH_HI))[0]
  # Guard against the case when an image has fewer than fg_rois_per_image
  # Foreground RoIs
  fg_rois_per_image = cfg.TEST.FG_NUMBER
  if cfg.TEST.BATCH_SIZE == -1:
    fg_rois_per_image = np.maximum(fg_inds.size, fg_rois_per_image)
  # 
  if fg_inds.size > 0:
    if fg_inds.size < fg_rois_per_image:
      while fg_inds.size < fg_rois_per_image:
        fg_inds = np.vstack((fg_inds, fg_inds))
      fg_inds = fg_inds.flatten()
  else:
    # copy primary region
    fg_inds = np.copy(fg_itself_inds)
    while fg_inds.size < fg_rois_per_image:
      fg_inds = np.vstack((fg_inds, fg_inds))
    fg_inds = fg_inds.flatten()
  # Sample foreground regions without replacement
  fg_inds = npr.choice(fg_inds, size=fg_rois_per_image,
                       replace=False)
  n_primary_region = fg_rois_per_image

  # 
  # Select the secondary region sets as background
  Zero = 0
  bg_rois_per_image = cfg.TEST.BG_NUMBER
  bg_flag = bg_rois_per_image > Zero
  # maybe we don't need to sample the background regions
  if bg_flag:
    # Select the secondary region sets as background as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = np.where((overlaps < cfg.TEST.BG_THRESH_HI) &
                       (overlaps >= cfg.TEST.BG_THRESH_LO))[0]
    # Sample background regions without replacement
    if bg_inds.size > 0:
      if bg_inds.size < bg_rois_per_image:
        while bg_inds.size < bg_rois_per_image:
          bg_inds = np.vstack((bg_inds, bg_inds))
        bg_inds = bg_inds.flatten()
      # 
      bg_rois_per_image = np.minimum(bg_rois_per_image,
                                          bg_inds.size)
      bg_inds = npr.choice(bg_inds, size=bg_rois_per_image,
                           replace=False)
      n_primary_region += bg_rois_per_image
    else:
      bg_flag = False

  # 
  # The indices that we're selecting (both fg and bg)
  # Place the primary region in the first place
  # Means the rois[0] is the primary region
  if bg_flag:
    keep_inds = np.append(fg_itself_inds, fg_inds)
    keep_inds = np.append(keep_inds, bg_inds)
  else:
    keep_inds = np.append(fg_itself_inds, fg_inds)

  # 
  # Only need the action label of primary region
  actionlabels = labels[fg_itself_inds]
  # But need the all region meeting the IoU, including the primary region and the secondary region set
  overlaps = overlaps[keep_inds]
  rois = rois[keep_inds]

  # 
  # Only need the primary bbox
  # we also can use this `__C.TEST.BBOX_THRESH` variable to make it
  # if we set `__C.TEST.BBOX_THRESH` to be `1.0`
  bbox_targets, bbox_loss_weights = \
          _get_bbox_regression_labels(roidb['bbox_targets'][fg_itself_inds, :],
                                      num_classes)
  if len(bbox_targets) != 1:
    print "missing primary regoin in \
        ${fast-rcnn-acton}/lib/action_roi_data_layer/minibatch \
        in `_sample_rois` function"
    sys.exit(1)

  # 
  # For debug visualizations
  # if you don't need, please comment it
  if has_visual_bbox:
    imagelist = [roidb['image'] ]
    imagelist = [il.split(os.sep)[-1] for il in imagelist]
    _vis_minibatch(im_blobs, rois, actionlabels, overlaps, imagelist)

  # 
  # where `rois` is bbox (x1, y1, x2, y2), 
  # bbox_targets is normalized regression target (nx, ny, nw, nh)
  # 
  n_rois_count_blob = [n_primary_region]
  n_rois_count_blob = np.array(n_rois_count_blob)

  # print "n_rois_count_blob:", n_rois_count_blob
  # print "n_rois_count_blob.shape:", n_rois_count_blob.shape

  blobs = {'data': im_blobs,
           'rois': rois,
           'labels': actionlabels,
           'n_rois_count': n_rois_count_blob,
           }

  if cfg.TEST.BBOX_REG:
    blobs['bbox_targets'] = bbox_targets_blob
    blobs['bbox_loss_weights'] = bbox_loss_blob

  return blobs



# Visualize
def _vis_minibatch(im_blobs, rois_blob, labels_blob, overlaps_blob, imagelist):
  """Visualize a mini-batch for debugging."""
  import matplotlib.pyplot as plt

  print
  print "im_blobs.shape:", im_blobs.shape
  print "rois_blob.shape[0]:", rois_blob.shape[0]
  print "labels_blob.shape:", labels_blob.shape
  print "overlaps_blob.shape:", overlaps_blob.shape

  for i in xrange(rois_blob.shape[0]):
    rois = rois_blob[i, :]
    # 
    im_ind = rois[0]
    roi = rois[1:]
    # 
    im = im_blobs[im_ind, :, :, :].transpose((1, 2, 0)).copy()
    im += cfg.PIXEL_MEANS
    im = im[:, :, (2, 1, 0)]
    im = im.astype(np.uint8)
    # 
    cls = labels_blob[0]
    plt.imshow(im)
    # 
    print 'idx:', i, ', im_ind:', im_ind, ', image path:', imagelist[0], \
        ', class:', cls, ', overlap: ', overlaps_blob[i], ", coors:", roi
    
    plt.gca().add_patch(
        plt.Rectangle((roi[0], roi[1]), roi[2] - roi[0],
                      roi[3] - roi[1], fill=False,
                      edgecolor='r', linewidth=3)
        )
    # 
    plt.show()


def _vis_secondary_region_with_class(im, primary_region, secondary_region, 
    pred_cls, gt_cls, output_dir = None):
  import matplotlib.pyplot as plt
  plt.imshow(im)
  plt.gca().add_patch(
      plt.Rectangle((primary_region[0], primary_region[1]), 
                     primary_region[2] - primary_region[0],
                     primary_region[3] - primary_region[1], 
                     fill=False, edgecolor='r', linewidth=3)
      )
  plt.gca().add_patch(
      plt.Rectangle((secondary_region[0], secondary_region[1]), 
                     secondary_region[2] - secondary_region[0],
                     secondary_region[3] - secondary_region[1], 
                     fill=False, edgecolor='g', linewidth=3)
      )
  # 
  sub_title = "gt: %s, pred: %s" % (gt_cls, pred_cls)
  plt.suptitle(sub_title)
  plt.show()



# 
# ##############################################################
# 


# 
def _bbox_pred(boxes, box_deltas):
  """Transform the set of class-agnostic boxes into class-specific boxes
  by applying the predicted offsets (box_deltas)
  """
  if boxes.shape[0] == 0:
      return np.zeros((0, box_deltas.shape[1]))

  boxes = boxes.astype(np.float, copy=False)
  widths = boxes[:, 2] - boxes[:, 0] + cfg.EPS
  heights = boxes[:, 3] - boxes[:, 1] + cfg.EPS
  ctr_x = boxes[:, 0] + 0.5 * widths
  ctr_y = boxes[:, 1] + 0.5 * heights

  dx = box_deltas[:, 0::4]
  dy = box_deltas[:, 1::4]
  dw = box_deltas[:, 2::4]
  dh = box_deltas[:, 3::4]

  pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
  pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
  pred_w = np.exp(dw) * widths[:, np.newaxis]
  pred_h = np.exp(dh) * heights[:, np.newaxis]

  pred_boxes = np.zeros(box_deltas.shape)
  # x1
  pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
  # y1
  pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
  # x2
  pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
  # y2
  pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h

  return pred_boxes


# 
def _clip_boxes(boxes, im_shape):
  """Clip boxes to image boundaries."""
  # x1 >= 0
  boxes[:, 0::4] = np.maximum(boxes[:, 0::4], 0)
  # y1 >= 0
  boxes[:, 1::4] = np.maximum(boxes[:, 1::4], 0)
  # x2 < im_shape[1]
  boxes[:, 2::4] = np.minimum(boxes[:, 2::4], im_shape[1] - 1)
  # y2 < im_shape[0]
  boxes[:, 3::4] = np.minimum(boxes[:, 3::4], im_shape[0] - 1)
  return boxes


# 
def vis_detections(im, class_name, dets, thresh=0.3):
  """Visual debugging of detections."""
  import matplotlib.pyplot as plt
  im = im[:, :, (2, 1, 0)]
  for i in xrange(np.minimum(10, dets.shape[0])):
      bbox = dets[i, :4]
      score = dets[i, -1]
      if score > thresh:
          plt.cla()
          plt.imshow(im)
          plt.gca().add_patch(
              plt.Rectangle((bbox[0], bbox[1]),
                            bbox[2] - bbox[0],
                            bbox[3] - bbox[1], fill=False,
                            edgecolor='g', linewidth=3)
              )
          plt.title('{}  {:.3f}'.format(class_name, score))
          plt.show()


# 
# ##############################################################
# 


#
# 
def im_detect(net, minibatch_db, num_classes, class_to_ind, \
    ind_to_class, has_visual_bbox = False, has_show_secondary_region = False):
  """Detect object classes in an image given object proposals.

  Arguments:
      net (caffe.Net): Fast R-CNN network to use
      im (ndarray): color image to test (in BGR order)
      boxes (ndarray): R x 4 array of object proposals

  Returns:
      scores (ndarray): R x K array of object class scores (K includes
          background as object category 0)
      boxes (ndarray): R x (4*K) array of predicted bounding boxes
  """
  # blobs: R * 5 (sx1, sy1, sx2, sy2, corresponding_scaled_index)
  # unused_im_scales: scale_num * 1 (the number of levels in pyramid)

  blobs = _get_minibatch(minibatch_db, num_classes, has_visual_bbox);

  # reshape network inputs
  net.blobs['data'].reshape(*(blobs['data'].shape))
  net.blobs['rois'].reshape(*(blobs['rois'].shape))
  net.blobs['labels'].reshape(*(blobs['labels'].shape)) 
  net.blobs['n_rois_count'].reshape(*(blobs['n_rois_count'].shape)) 
  # print
  # print "blobs['data'].shape:", blobs['data'].shape
  # print "blobs['rois'].shape:", blobs['rois'].shape
  # print "blobs['labels'].shape:", blobs['labels'].shape
  # print "blobs['n_rois_count'].shape:", blobs['n_rois_count'].shape
  # print

  # 
  out_names_blobs = ["primary_score", "secondary_score", \
      "secondary_max_score", "cls_score", "cls_prob", "accuracy"]
  blobs_out = net.forward(out_names_blobs,
                          data=blobs['data'].astype(np.float32, copy=False),
                          rois=blobs['rois'].astype(np.float32, copy=False),
                          labels=blobs['labels'].astype(np.float32, copy=False),
                          n_rois_count=blobs['n_rois_count'].astype(np.float32, copy=False))


  # # use fc scores
  cls_score = net.blobs['cls_score'].data
  # # use softmax estimated probabilities
  cls_prob = blobs_out['cls_prob']

  pred_cls_ind = np.argmax(cls_prob)
  gt_cls_ind = blobs['labels'][0]
  cls = blobs_out['accuracy']

  secondary_score = blobs_out['secondary_score']
  primary_score = blobs_out['primary_score']
  # sr_ind = secondary_score.argmax(axis=0)
  sr_ind = []
  secondary_max_score2 = []
  for nc in xrange(num_classes):
    sind = np.argmax(secondary_score[:, nc])
    smax = secondary_score[sind, nc]
    sr_ind.append(sind)
    secondary_max_score2.append(smax)
  secondary_max_score = blobs_out['secondary_max_score']   

  # has_show_secondary_region = True
  if has_show_secondary_region:
    im_ind = blobs['rois'][0][0]
    # print "im_ind:", im_ind
    # print "data.shape:", blobs['data'].shape
    im = blobs['data'][im_ind, :, :, :].transpose((1, 2, 0)).copy()
    im += cfg.PIXEL_MEANS
    im = im[:, :, (2, 1, 0)]
    im = im.astype(np.uint8)
    primary_region = blobs['rois'][0][1:]
    sind = sr_ind[pred_cls_ind]
    # *******************************************************************************
    # if use pyramid, the level of the secondary_region may match the level of primary
    # *******************************************************************************
    secondary_region = blobs['rois'][sind][1:]

    _vis_secondary_region_with_class(im, primary_region, secondary_region, \
        ind_to_class[pred_cls_ind], ind_to_class[gt_cls_ind])

  # print
  # print "secondary_score.shape:"
  # print secondary_score.shape
  # print
  # print "secondary_score:"
  # print secondary_score
  # print
  # print "sr_ind:"
  # print sr_ind
  # print
  # print "secondary_max_score:"
  # print secondary_max_score
  # print
  # print "secondary_max_score2:"
  # print secondary_max_score2
  # print
  # print "primary_score:"
  # print primary_score
  # print
  # print "cls_score:"
  # print cls_score
  # print 
  # print "cls_prob:"
  # print cls_prob
  # print 
  # print "************************************************"
  # # sys.exit(1)

  # print "groundtruth cls:", ind_to_class[gt_cls_ind], "(", gt_cls_ind, ")"
  # print "predicted cls:", ind_to_class[pred_cls_ind], "(", pred_cls_ind, ")"
  # print "predicted accuracy:", cls

  boxes = minibatch_db[0]['boxes']
  if cfg.TEST.BBOX_REG:
    # Apply bounding-box regression deltas
    box_deltas = blobs_out['bbox_pred']
    pred_boxes = _bbox_pred(boxes, box_deltas)
    pred_boxes = _clip_boxes(pred_boxes, im.shape)
  else:
    # Simply repeat the boxes, once for each class
    pred_boxes = np.tile(boxes, (1, cls_prob.shape[1]))

  return cls_prob, pred_boxes, pred_cls_ind, gt_cls_ind



# 
# test images
def test_net(net, roidb, output_dir, num_images, \
      num_classes, class_to_ind, ind_to_class, \
      bbox_means = None, bbox_stds = None, \
      has_show_secondary_region = False):
  """Test a Fast R-CNN network on an image database."""
  # timers
  _t = {'im_detect' : Timer(), 'misc' : Timer()}
  boxes_blob = []
  scores_blob = []
  has_visual_bbox = False # True, False

  accuracy_counts = np.zeros((num_classes, num_classes), dtype=np.float32)
  cls_count = np.zeros((num_classes, ), dtype=np.float32)

  _t['im_detect'].tic()
  ## #########################################################
  ## Here we don't need any `detection`, so we command these codes below
  for i in xrange(num_images):
    print "process the", i, "image (", num_images, " total images)"
    minibatch_db = []
    minibatch_db.append(roidb[i])
    # 
    scores, boxes, pred_cls_ind, gt_cls_ind = im_detect(\
        net, minibatch_db, num_classes, \
        class_to_ind, ind_to_class, \
        has_visual_bbox=has_visual_bbox, \
        has_show_secondary_region=has_show_secondary_region)
    # 
    cls_count[gt_cls_ind] += 1
    accuracy_counts[gt_cls_ind][pred_cls_ind] += 1

    # boxes_blob.append(boxes)
    # scores_blob.append(scores)

  # 
  _draw_confusion_matrix(num_classes, accuracy_counts, cls_count, \
      output_dir, class_to_ind, ind_to_class)

  # 
  _t['im_detect'].toc()
  print 'im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
        .format(i + 1, num_images, _t['im_detect'].average_time,
                _t['misc'].average_time)
  #
  # #########################################################               
  # 


def _draw_confusion_matrix(num_classes, accuracy_counts, cls_count, \
    output_dir, class_to_ind, ind_to_class):
  # ###########################################################
  # draw the results, except background
  for ni in range(1, num_classes):
    accuracy_counts[ni] /= cls_count[ni]
  # 
  total_accuracy = 0
  accuracy_list = []
  for ni in range(1, num_classes):
    total_accuracy += accuracy_counts[ni][ni]
    accuracy_list.append(accuracy_counts[ni][ni])
  total_accuracy /= (num_classes - 1)
  accuracy_list.append(total_accuracy)
  
  print
  print "accuracy are described below:"
  for ni in range(1, num_classes):
    print ind_to_class[ni], ":", accuracy_list[ni - 1]
  print "total :", accuracy_list[num_classes - 1] 
  print
  # 
  fig = plt.figure()
  plt.clf()
  ax = fig.add_subplot(111)
  ax.set_aspect(1)
  res = ax.imshow(np.array(accuracy_counts), cmap=plt.cm.jet, 
                  interpolation='nearest')
  width = num_classes
  height = num_classes
  # 
  for idx1 in xrange(width):  # x
    for idx2 in xrange(height): # y
      ax.annotate(str(accuracy_counts[idx1][idx2]), xy=(idx2, idx1), 
                  horizontalalignment='center',
                  verticalalignment='center')

  cb = fig.colorbar(res)
  ckeys = ind_to_class.keys()
  ckeys.sort()
  classes = []
  for ck in ckeys:
    classes.append(ind_to_class[ck])
  
  plt.xticks(range(width), classes[:width])
  plt.yticks(range(height), classes[:height])
  plt.show()
  fig_confusion_matrix_file = os.path.join(output_dir, 'confusion_matrix.png')
  # plt.savefig(fig_confusion_matrix_file, format='png')