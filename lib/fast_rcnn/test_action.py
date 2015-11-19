# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# 
# Modified by DDK
# --------------------------------------------------------

"""Test a Fast R-CNN network on an imdb.roidb (image database)."""

import caffe

# for action recognition task in still/static images
from utils.cython_nms import nms
from fast_rcnn.config_action import cfg, global_dict, get_output_dir
from utils.blob import im_list_to_blob
# 
import cv2
import heapq
import os, sys
import cPickle
import argparse
import numpy as np
from utils.timer import Timer
from numpy import random as npr
import matplotlib.pyplot as plt

_bbox_means, _bbox_stds = None, None
_class_to_ind, _ind_to_class = None, None


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
  # Here treat the pyramid of one image as a batch
  blob = im_list_to_blob(processed_ims)
  return blob, np.array(im_scales)


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
    levels = diff_areas.argmin(axis=1)[:, np.newaxis]
  else:
    levels = np.zeros((im_rois.shape[0], 1), dtype=np.int)

  # scale
  if cfg.TEST.HAS_CONTROL_THE_SAME_SCALE:
    # get the level of primary region -- primary region is placed in the first place
    primary_level = levels[0]
    # the secondary regions must match the scale of the primary region
    levels = np.ones((im_rois.shape[0], 1), dtype=np.int)
    levels = levels * primary_level
  
  rois = im_rois * scales[levels]
  return rois, levels


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


# here, number of minibatch must be 1
def _get_minibatch(roidb, num_classes):
  """Given a roidb, construct a minibatch sampled from it."""
  # 
  num_images = len(roidb)
  assert(num_images, 1)
  roidb = roidb[0]

  # Get image pyramid and corresponding scales and 
  # indices of scales(recording the suited scale `224 * 224`)
  im = cv2.imread(roidb['image'])
  boxes = roidb['boxes']
  im_blobs, im_scales = _get_image_blob(im)

  # Sample primary region and the secordary regions
  # label = class RoI has max overlap with
  labels = roidb['max_classes']
  overlaps = roidb['max_overlaps']
  rois = roidb['boxes']
  rois = _get_rois_blob(rois, im_scales)

  # Select the primary region
  n_primary_region = 1
  fg_itself_ind = np.where(overlaps >= cfg.TEST.SR_ITSELF_THRESH)[0]
  if fg_itself_ind.size > n_primary_region:
    fg_itself_ind = npr.choice(fg_itself_ind, size=n_primary_region,
                         replace=False)
  if fg_itself_ind.size != n_primary_region \
      or overlaps[fg_itself_ind[0]] != cfg.TEST.FG_ITSELF_THRESH:
    sys.exit(1)
  primary_region = rois[fg_itself_ind]
  primary_level = primary_region[0][0]
  primary_scale = im_scales[primary_level]

  # Select the secondary regions if have
  n_scene_region = 0
  scene_region = None
  n_fg_secondary_regions = 0
  fg_secondary_regions = None
  n_bg_secondary_regions = 0
  bg_secondary_regions = None
  if cfg.TEST.HAS_SECONDARY_REGIONS:
    # Select the whole image as a secondary region
    # (height, width, channels) mapping to (x1, y1, x2, y2)
    if cfg.TEST.NEED_WHOLE_IMAGE_BBOX:
      im_shape = im.shape
      sh = im_shape[0] * primary_scale
      sw = im_shape[1] * primary_scale
      scene_region = [primary_level, 0, 0, sw - 1, sh - 1]
      n_scene_region = 1

    if cfg.TEST.HAS_FG_SECONDARY_REGIONS:
      # Select the secondary region sets as foreground, e.g. [0.5, 0.75]
      fg_inds = np.where((overlaps >= cfg.TEST.SR_FG_THRESH_LO) & 
              (overlaps <= cfg.TEST.SR_FG_THRESH_HI))[0]
      # Foreground RoIs
      if fg_inds.size <= 0 and cfg.TEST.SR_FG_COPY_PER_NUMBER > 0:
        print "copy primary region"
        fg_inds = np.copy(fg_itself_ind)
        while fg_inds.size < cfg.TEST.SR_FG_COPY_PER_NUMBER:
          fg_inds = np.vstack((fg_inds, fg_inds))
        fg_inds = fg_inds.flatten()
        fg_inds = npr.choice(fg_inds, size=cfg.TEST.SR_FG_COPY_PER_NUMBER,
                             replace=False)
      if fg_inds.size > 0:
        fg_secondary_regions = rois[fg_inds]
        n_fg_secondary_regions = fg_inds.size

    if cfg.TEST.HAS_BG_SECONDARY_REGIONS:
       # Select the secondary region sets as background, e.g. [0.1, 0.45]
      bg_inds = np.where((overlaps >= cfg.TEST.SR_BG_THRESH_LO) & 
              (overlaps <= cfg.TEST.SR_BG_THRESH_HI))[0]
      if bg_inds.size > 0:
        bg_secondary_regions = rois[bg_inds]
        n_bg_secondary_regions = bg_inds.size


  # The indices that we're selecting (both fg and bg)
  # Place the primary region in the first place
  # Means the rois[0] is the primary region
  primary_overlap = overlaps[fg_itself_ind]

  if scene_region is not None:
    # deal with overlaps
    primary_w = primary_region[0][2] - primary_region[0][0] + 1.0
    primary_h = primary_region[0][3] - primary_region[0][1] + 1.0
    scene_w = scene_region[2] + 1.0
    scene_h = scene_region[3] + 1.0
    scene_overlap = (primary_w * primary_h) / (scene_w * scene_h)
    primary_overlap = np.append(primary_overlap, scene_overlap)
    # deal with regions/proposals
    # add primary_region and scene_region
    primary_region = np.vstack((primary_region, scene_region))

  if fg_secondary_regions is not None:
    # deal with overlaps
    fg_secondary_overlap = overlaps[fg_inds]
    primary_overlap = np.append(primary_overlap, fg_secondary_overlap)
    primary_region = np.vstack((primary_region, fg_secondary_regions))

  if bg_secondary_regions is not None:
    bg_secondary_overlap = overlaps[bg_inds]
    primary_overlap = np.append(primary_overlap, bg_secondary_overlap)
    primary_region = np.vstack((primary_region, bg_secondary_regions))


  # Place the primary region in the first place
  # But need the all region meeting the IoU, 
  # including the primary region and the secondary region set
  rois = primary_region
  # Only need the action label of primary region
  actionlabels = labels[fg_itself_ind]
  # Get the overlaps
  overlaps = primary_overlap

 
  # Only need the primary bbox
  bbox_targets, bbox_loss_weights = \
          _get_bbox_regression_labels(roidb['bbox_targets'][fg_itself_ind, :],
                                      num_classes)

  if cfg.TEST.HAS_CONTROL_THE_SAME_SCALE:
    print "primary_level:", primary_level
    im_blobs = im_blobs[primary_level]
    im_blobs = [im_blobs]
    im_blobs = np.array(im_blobs)
    im_scales = im_scales[primary_level]
    im_scales = [im_scales]
    im_scales = np.array(im_scales)

  # Visualizations
  has_visual_bbox = cfg.TEST.HAS_VISUAL_BBOX
  if has_visual_bbox:
    imagelist = [roidb['image'] ]
    imagelist = [il.split(os.sep)[-1] for il in imagelist]
    _vis_minibatch(im_blobs, rois, actionlabels, overlaps, imagelist)
  
  # Set the data/labels for DataLayer
  blobs = {'data': im_blobs, 'rois': rois, 'labels': actionlabels}
  if cfg.TEST.BBOX_REG:
    blobs['bbox_targets'] = bbox_targets_blob
    blobs['bbox_loss_weights'] = bbox_loss_blob

  return blobs, im_scales


# Visualize
def _vis_minibatch(im_blobs, rois_blob, labels_blob, overlaps_blob, imagelist):
  """Visualize a mini-batch for debugging."""
  import matplotlib.pyplot as plt

  print
  print "im_blobs.shape:", im_blobs.shape
  print "rois_blob.shape[0]:", rois_blob.shape[0]
  print "labels_blob.shape:", labels_blob.shape
  print "overlaps_blob.shape:", overlaps_blob.shape

  rb_len = rois_blob.shape[0]
  if cfg.TEST.VISUAL_BBOX_LEN > 0:
    rb_len = min(cfg.TEST.VISUAL_BBOX_LEN, rb_len)

  for i in xrange(rb_len):
    rois = rois_blob[i, :]
    im_ind = rois[0]
    roi = rois[1:]
    im_ind2 = im_ind
    if cfg.TEST.HAS_CONTROL_THE_SAME_SCALE:
      im_ind2 = 0
    im = im_blobs[im_ind2, :, :, :].transpose((1, 2, 0)).copy()
    im += cfg.PIXEL_MEANS
    im = im[:, :, (2, 1, 0)]
    im = im.astype(np.uint8)
    cls = labels_blob[0]
    plt.imshow(im)
    print 'idx:', i, ', im_ind:', im_ind, ', image path:', imagelist[0], \
        ', class:', cls, ', overlap: ', overlaps_blob[i], ", coors:", roi
    # 
    plt.gca().add_patch(
        plt.Rectangle((roi[0], roi[1]), roi[2] - roi[0],
                      roi[3] - roi[1], fill=False,
                      edgecolor='r', linewidth=3)
        )
    # 
    plt.show()


# 
def _vis_secondary_region_with_class(im, primary_region, secondary_region, 
    pred_cls, gt_cls, image_output_dir , imagename, img_ext):
  import matplotlib.pyplot as plt
  plt.imshow(im)
  # assert(primary_region)
  plt.gca().add_patch(
      plt.Rectangle((primary_region[0], primary_region[1]), 
                     primary_region[2] - primary_region[0],
                     primary_region[3] - primary_region[1], 
                     fill=False, edgecolor='r', linewidth=3)
      )
  # If have the max-selected secondary region
  if secondary_region is not None:
    plt.gca().add_patch(
        plt.Rectangle((secondary_region[0], secondary_region[1]), 
                       secondary_region[2] - secondary_region[0],
                       secondary_region[3] - secondary_region[1], 
                       fill=False, edgecolor='g', linewidth=3)
        )
  # Show the results as the tile form
  sub_title = "gt: %s, pred: %s" % (gt_cls, pred_cls)
  plt.suptitle(sub_title)
  # Save the image
  if image_output_dir:
    imagepath = image_output_dir + os.sep + imagename + \
        "_%s_%s" % (gt_cls, pred_cls) + img_ext
    plt.savefig(imagepath)
  else:
    plt.show()
  # Clear the figure handler
  plt.clf()


# 
# ##############################################################
# 


# each time process one image
def im_detect(net, minibatch_db, num_classes, output_dir=None):
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
  global _ind_to_class, _class_to_ind
  assert(_ind_to_class)
  assert(_class_to_ind)
  # 
  assert(len(minibatch_db), 1), "each time, we process one image"
  # Get the input data/labels
  blobs, im_scales = _get_minibatch(minibatch_db, num_classes);

  if cfg.TEST.HAS_SECONDARY_REGIONS:
    blobs_size = len(blobs['rois']) - 1
    sr_per_num = cfg.TEST.SR_PER_NUMBER - 1
    sr_iter_count = blobs_size / sr_per_num
    if blobs_size % sr_per_num:
      sr_iter_count += 1
    out_names_blobs = ["primary_score", "secondary_score", \
        "secondary_max_score", "cls_score", "cls_prob", "accuracy"]
  else:
    sr_iter_count = 1
    out_names_blobs = ["primary_score",]
    
  # 
  data_blobs = blobs['data']
  label_blob = blobs['labels']
  primary_roi_blob = blobs['rois'][0, :]
  secordary_rois_blobs = None
  if cfg.TEST.HAS_SECONDARY_REGIONS:
    secordary_rois_blobs = blobs['rois'][1:, :]
  primary_score = None
  secondary_score = None
  # 
  print "process", len(blobs['rois']), "boxes each image"

  for sic in xrange(sr_iter_count):
    if cfg.TEST.HAS_SECONDARY_REGIONS:
      start_idx = sic * sr_per_num
      end_idx = (sic + 1) * sr_per_num
      end_idx = min(blobs_size, end_idx) 
      # Get input info
      rois_blobs = secordary_rois_blobs[start_idx: end_idx, :]
      rois_blobs = np.vstack((primary_roi_blob, rois_blobs))
      # Determine the number of secondary regions
      n_rois_count = [end_idx - start_idx]
      n_rois_count_blob = np.array(n_rois_count)
    else:
      rois_blobs = primary_roi_blob
      rois_blobs = np.array(rois_blobs)
      n_rois_count = [0]
      n_rois_count_blob = np.array(n_rois_count)

    # Reshape network inputs
    net.blobs['data'].reshape(*(data_blobs.shape))
    net.blobs['rois'].reshape(*(rois_blobs.shape))
    net.blobs['labels'].reshape(*(label_blob.shape)) 
    net.blobs['n_rois_count'].reshape(*(n_rois_count_blob.shape)) 

    # Forward
    blobs_out = net.forward(out_names_blobs,
                            data=data_blobs.astype(np.float32, copy=False),
                            rois=rois_blobs.astype(np.float32, copy=False),
                            labels=label_blob.astype(np.float32, copy=False),
                            n_rois_count=n_rois_count_blob.astype(np.float32, copy=False))
    # Get the primary region's scores for each class
    if primary_score is None:
      primary_score = blobs_out['primary_score']
      primary_score = primary_score[0]      
    # 
    if cfg.TEST.HAS_SECONDARY_REGIONS:
      if secondary_score is None:
        secondary_score = blobs_out['secondary_score']
      else:
        secondary_score = np.vstack((secondary_score, blobs_out['secondary_score']))

  cls_score = []
  sr_ind = []
  secondary_max_score = []
  # 
  if cfg.TEST.HAS_SECONDARY_REGIONS:
    for nc in xrange(num_classes):
      sind = np.argmax(secondary_score[:, nc])
      smax = np.max(secondary_score[:, nc])
      sr_ind.append(sind)
      secondary_max_score.append(smax)
    for nc in xrange(num_classes):
      nc_score = primary_score[nc] + secondary_max_score[nc]
      cls_score.append(nc_score)
  else:
    cls_score = primary_score
  # 
  cls_prob = np.exp(cls_score)
  cls_prob = cls_prob / np.sum(cls_prob)
  pred_cls_ind = np.argmax(cls_prob)
  gt_cls_ind = blobs['labels'][0]

  # 
  if cfg.TEST.HAS_SHOW_REGION:
    print "predicted cls:", _ind_to_class[pred_cls_ind], "(", pred_cls_ind, ")"
    print "groundtruth cls:", _ind_to_class[gt_cls_ind], "(", gt_cls_ind, ")"
    print
    # get the original image
    imagepath = minibatch_db[0]['image']
    im = cv2.imread(imagepath)    
    # Get the primary region's corresponding scaled index
    prlevel = blobs['rois'][0][0]
    primary_region = blobs['rois'][0][1:]
    if cfg.TEST.HAS_CONTROL_THE_SAME_SCALE:
      prlevel = 0
    primary_region = primary_region / im_scales[prlevel]

    # 
    imagename = imagepath.split("/")[-1]
    imagename, img_ext = imagename.split(".")
    img_ext = "." + img_ext
    image_output_dir = os.path.join(output_dir, cfg.TEST.PRIMARY_SECONDARY_VISUAL)
    # Build the directory
    if not os.path.exists(image_output_dir):
      os.makedirs(image_output_dir)
    # 
    pred_cls = _ind_to_class[pred_cls_ind]
    gt_cls = _ind_to_class[gt_cls_ind]

    # 
    has_secondary_regions = cfg.TEST.HAS_SECONDARY_REGIONS
    has_show_secondary_region = cfg.TEST.HAS_SHOW_SECONDARY_REGION
    secondary_region = None
    if has_secondary_regions and has_show_secondary_region:
      # 
      sind = sr_ind[pred_cls_ind]
      srlevel = blobs['rois'][sind][0]
      if cfg.TEST.HAS_CONTROL_THE_SAME_SCALE:
        srlevel = 0
      secondary_region = blobs['rois'][sind][1:]
      secondary_region = secondary_region / im_scales[srlevel]
  
    _vis_secondary_region_with_class(im, primary_region, secondary_region, \
        pred_cls, gt_cls, image_output_dir, imagename, img_ext)
    print
    print "*******************************************************************"
    print

  # 
  return cls_prob, pred_cls_ind, gt_cls_ind



# 
# test images
def test_net(net, roidb, output_dir, num_images, num_classes):
  """Test a Fast R-CNN network on an image database."""
  # 
  global _bbox_means, _bbox_stds, _ind_to_class, _class_to_ind
  _bbox_means = global_dict[cfg.TEST_BBOX_MEANS]
  _bbox_stds = global_dict[cfg.TEST_BBOX_STDS]
  _class_to_ind = global_dict[cfg.CLASS_TO_IND]
  _ind_to_class = global_dict[cfg.IND_TO_CLASS]

  # timers
  _t = {'im_detect' : Timer(), 'misc' : Timer()}
  # 
  boxes_blob = []
  scores_blob = []
  has_visual_bbox = cfg.TEST.HAS_VISUAL_BBOX
  accuracy_counts = np.zeros((num_classes, num_classes), dtype=np.float32)
  cls_count = np.zeros((num_classes, ), dtype=np.float32)

  # 
  _t['im_detect'].tic()
  image_index_len = num_images
  if cfg.LESSEN_DEBUG_TIME:
    image_index_len = min(image_index_len, cfg.LESSEN_DEBUG_IMAGE_INDEX_LEN)
  
  for i in xrange(image_index_len):
    print "process the", i + 1, "image (", image_index_len, " total images)"
    minibatch_db = []
    minibatch_db.append(roidb[i])
    # 
    scores, pred_cls_ind, gt_cls_ind = im_detect(net, \
        minibatch_db, num_classes, output_dir=output_dir)
    # 
    cls_count[gt_cls_ind] += 1
    accuracy_counts[gt_cls_ind][pred_cls_ind] += 1
  # 
  _draw_confusion_matrix(num_classes, accuracy_counts, cls_count, output_dir)
  # 
  _t['im_detect'].toc()
  print 'im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
        .format(i + 1, num_images, _t['im_detect'].average_time,
                _t['misc'].average_time)
  


def _draw_confusion_matrix(num_classes, accuracy_counts, cls_count, output_dir):
  global _ind_to_class, _class_to_ind
  assert(_ind_to_class)
  assert(_class_to_ind)
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
  

  # 
  accuracy_file = os.path.join(output_dir, "accuracy.log")
  af_hd = open(accuracy_file, "w")
  print
  print "accuracy are described below:"
  for ni in range(1, num_classes):
    str_info = _ind_to_class[ni] + ": " + str(accuracy_list[ni - 1])
    af_hd.write(str_info + "\n")
    print str_info
  str_info = "total :" +  str(accuracy_list[num_classes - 1])
  print str_info
  af_hd.write(str_info + "\n")

  # write the confusion matrix
  padding = 16
  str_info = ""
  for idx in xrange(num_classes):
    if idx == 0:
      continue 
    str_info = str_info + _ind_to_class[idx].ljust(padding)
  for idx1 in xrange(num_classes): 
    if idx1 == 0:
      continue
    str_info = _ind_to_class[idx].ljust(padding)
    for idx2 in xrange(num_classes): 
      if idx2 == 0:
        continue
      str_info = str_info + str(accuracy_counts[idx1][idx2]).ljust(padding)
    af_hd.write(str_info + "\n")
  af_hd.write(str_info + "\n")
  af_hd.close()
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
  ckeys = _ind_to_class.keys()
  ckeys.sort()
  classes = []
  for ck in ckeys:
    classes.append(_ind_to_class[ck])
  plt.xticks(range(width), classes[:width])
  plt.yticks(range(height), classes[:height])
  # plt.show()
  fig_confusion_matrix_file = os.path.join(output_dir, 'confusion_matrix.png')
  plt.savefig(fig_confusion_matrix_file, format='png')
  plt.clf()

