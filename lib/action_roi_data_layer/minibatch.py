# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# 
# Modified by ddk
# --------------------------------------------------------

"""Compute minibatch blobs for training a Fast R-CNN network."""

import os, sys
import numpy as np
import numpy.random as npr
import cv2
from fast_rcnn.config_action import cfg
from utils.blob import prep_im_for_blob, im_list_to_blob



def _get_image_blob(roidb, scale_inds):
  """
  Builds an input blob from the images in the roidb at the specified
  scales.
  """
  num_images = len(roidb)
  processed_ims = []
  im_scales = []
  im_shapes = []

  for i in xrange(num_images):
    im = cv2.imread(roidb[i]['image'])
    # Check flipped or not
    if roidb[i]['flipped']:
      im = im[:, ::-1, :]
    # record the shape of origin image: (height, width, channels)
    im_shapes.append(im.shape)

    target_size = cfg.TRAIN.SCALES[scale_inds[i]]
    im, im_scale = prep_im_for_blob(im, cfg.PIXEL_MEANS, target_size,
                                    cfg.TRAIN.MAX_SIZE)
    im_scales.append(im_scale)
    processed_ims.append(im)

  # Create a blob to hold the input images
  blob = im_list_to_blob(processed_ims)

  return blob, im_scales, im_shapes



def _sample_rois(roidb, img_shape, rois_per_image, fg_rois_per_image, num_classes):
  """
  Generate a random sample of RoIs comprising foreground and background
  examples.
  """
  # label = class RoI has max overlap with
  labels = roidb['max_classes']
  overlaps = roidb['max_overlaps']
  rois = roidb['boxes']
  assert((rois_per_image > 0))

  # Select the primary region
  n_primary_region = 1
  fg_itself_inds = np.where(overlaps == cfg.TRAIN.FG_ITSELF_THRESH)[0]
  if fg_itself_inds.size > n_primary_region:
    fg_itself_inds = npr.choice(fg_itself_inds, size=n_primary_region,
                         replace=False)
  if fg_itself_inds.size != 1 or overlaps[fg_itself_inds[0]] != cfg.TRAIN.FG_ITSELF_THRESH:
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
  # ###################################################################
  # 

  bg_flag = None
  fg_inds = None
  keep_inds = None
  scene_region = None
  # 
  Zero = 0
  n_scene_region = 0
  if cfg.TRAIN.HAS_SECONDARY_REGIONS:
    # (height, width, channels) mapping to (x1, y1, x2, y2)
    if cfg.TRAIN.NEED_WHOLE_IMAGE_BBOX:
      scene_region = [0, 0, img_shape[1] - 1, img_shape[0] - 1]
      n_scene_region = 1
      fg_rois_per_image = fg_rois_per_image - n_scene_region
      assert((rois_per_image > 1))
    # 
    # ###################################################################
    # foreground secondary regions/proposals
    if fg_rois_per_image > Zero and rois_per_image >= fg_rois_per_image:
      # Select the secondary region sets as foreground
      fg_inds = np.where((overlaps >= cfg.TRAIN.FG_THRESH_LO) & 
              (overlaps <= cfg.TRAIN.FG_THRESH_HI))[0]
      # Guard against the case when an image has fewer than fg_rois_per_image
      # Foreground RoIs
      if fg_inds.size > 0:
        # copy fg rois itself
        if fg_inds.size < fg_rois_per_image:
          while fg_inds.size < fg_rois_per_image:
            fg_inds = np.vstack((fg_inds, fg_inds))
          fg_inds = fg_inds.flatten()
      # copy primary region
      else:
        fg_inds = np.copy(fg_itself_inds)
        while fg_inds.size < fg_rois_per_image:
          fg_inds = np.vstack((fg_inds, fg_inds))
        fg_inds = fg_inds.flatten()
      # 
      # Sample foreground regions without replacement
      fg_rois_per_this_image = np.minimum(fg_rois_per_image, fg_inds.size)
      assert(fg_rois_per_image, fg_rois_per_this_image), \
          'fg_rois_per_image ({}) must be equal to fg_rois_per_this_image ({})'. \
          format(fg_rois_per_image, fg_rois_per_this_image)
      if fg_inds.size > 0:
        fg_inds = npr.choice(fg_inds, size=fg_rois_per_this_image,
                             replace=False)
      # 
      # ###################################################################
      # background secondary regions/proposals
      bg_rois_per_this_image = rois_per_image - \
          fg_rois_per_this_image - n_primary_region - n_scene_region
      bg_flag = bg_rois_per_this_image > Zero
      # maybe we don't need to sample the background regions
      if bg_flag:
        # Select the secondary region sets as background as those within [BG_THRESH_LO, BG_THRESH_HI)
        bg_inds = np.where((overlaps < cfg.TRAIN.BG_THRESH_HI) &
                           (overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
        # Compute number of background RoIs to take from this image 
        # guarding against there being fewer than desired
        # Sample foreground regions without replacement
        if bg_inds.size > 0:
          if bg_inds.size < bg_rois_per_this_image:
            while bg_inds.size < bg_rois_per_this_image:
              bg_inds = np.vstack((bg_inds, bg_inds))
            bg_inds = bg_inds.flatten()
          # 
          bg_rois_per_this_image = np.minimum(bg_rois_per_this_image,
                                              bg_inds.size)
          bg_inds = npr.choice(bg_inds, size=bg_rois_per_this_image,
                               replace=False)
        else:
          # select regions from foreground ignoring background regions
          bg_inds = np.copy(fg_inds)
          while bg_inds.size < bg_rois_per_this_image:
            bg_inds = np.vstack((bg_inds, bg_inds))
          bg_inds = bg_inds.flatten()
          # 
          bg_rois_per_this_image = np.minimum(bg_rois_per_this_image,
                                              bg_inds.size)
          bg_inds = npr.choice(bg_inds, size=bg_rois_per_this_image,
                               replace=False)
      # 
      # ###################################################################
      # Get the secondary regions meeting the conditions
      # The indices that we're selecting (both fg and bg)
      keep_inds = fg_inds
      if bg_flag:
        keep_inds = np.append(keep_inds, bg_inds)

  # 
  # #################################################################################
  # 

  # The indices that we're selecting (both fg and bg)
  # Place the primary region in the first place
  # Means the rois[0] is the primary region
  n_rois_count = 0
  primary_region = rois[fg_itself_inds]
  primary_overlap = overlaps[fg_itself_inds]
  # print "primary_overlap:", primary_overlap
  # print "primary_overlap.shape:", primary_overlap.shape
  # print "primary_region:", primary_region
  # print "primary_region.shape:", primary_region.shape

  if scene_region is not None:
    # deal with overlaps
    primary_w = primary_region[0][2] - primary_region[0][0] + 1.0
    primary_h = primary_region[0][3] - primary_region[0][1] + 1.0
    scene_w = scene_region[2] + 1.0
    scene_h = scene_region[3] + 1.0
    scene_overlap = (primary_w * primary_h) / (scene_w * scene_h)
    # print "scene_overlap:", scene_overlap
    primary_overlap = np.append(primary_overlap, scene_overlap)
    # print "primary_overlap&scene_overlap:", primary_overlap
    # print "primary_overlap&scene_overlap.shape:", primary_overlap.shape
    # 
    # deal with regions/proposals
    # add primary_region and scene_region
    primary_region = np.vstack((primary_region, scene_region))
    # print "scene_region:", scene_region
    # print "primary_region&scene_region:", primary_region
    # print "primary_region&scene_region.shape:", primary_region.shape
    # 
    # record the number of secondary regions/proposals
    n_rois_count += 1

  if keep_inds is not None:
    # deal with overlaps
    secondary_overlap = overlaps[keep_inds]
    # print "secondary_overlap:", secondary_overlap
    # print "secondary_overlap.shape:", secondary_overlap.shape
    primary_overlap = np.append(primary_overlap, secondary_overlap)
    # print "primary_overlap&scene_overlap&secondary_overlap:", primary_overlap
    # print "primary_overlap&scene_overlap&secondary_overlap.shape:", primary_overlap.shape
    # 
    # deal with regions/proposals
    # add primary_region and scene_region
    secondary_regions = rois[keep_inds]
    # print "secondary_regions:", secondary_regions
    # print "secondary_regions.shape:", secondary_regions.shape
    primary_region = np.vstack((primary_region, secondary_regions))
    # print "primary_region&scene_region&secondary_regions:", primary_region
    # print "primary_region&scene_region&secondary_regions.shape:", primary_region.shape
    # 
    # record the number of secondary regions/proposals
    n_rois_count += keep_inds.size
    # print "n_rois_count:", n_rois_count 


  # Place the primary region in the first place
  # But need the all region meeting the IoU, 
  # including the primary region and the secondary region set
  rois = primary_region
  # Only need the action label of primary region
  actionlabels = labels[fg_itself_inds]
  # Get the overlaps
  overlaps = primary_overlap
  # Get the number of secondary regions, 
  # including scene, foreground and background, if have any
  n_rois_count = [n_rois_count]

  
  # Only need the primary bbox
  bbox_targets, bbox_loss_weights = \
          _get_bbox_regression_labels(roidb['bbox_targets'][fg_itself_inds, :],
                                      num_classes)
  if len(bbox_targets) != 1:
    print "missing primary regoin in \
        ${fast-rcnn-acton}/lib/action_roi_data_layer/minibatch \
        in `_sample_rois` function"
    sys.exit(1)

  return rois, actionlabels, overlaps, n_rois_count, bbox_targets, bbox_loss_weights


# rescale the position (x1, y1, x2, y2)
def _project_im_rois(im_rois, im_scale_factor):
  """Project image RoIs into the rescaled training image."""
  rois = im_rois * im_scale_factor
  return rois



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
def get_minibatch(roidb, num_classes, has_visual_bbox = False):
  """Given a roidb, construct a minibatch sampled from it."""
  num_images = len(roidb)
  # Sample random scales to use for each image in this batch
  random_scale_inds = npr.randint(0, high=len(cfg.TRAIN.SCALES),
                                  size=num_images)
  assert(cfg.TRAIN.BATCH_SIZE % num_images == 0), \
      'num_images ({}) must divide BATCH_SIZE ({})'. \
      format(num_images, cfg.TRAIN.BATCH_SIZE)
  # Get the input image blob, formatted for caffe
  im_blob, im_scales, im_shapes = _get_image_blob(roidb, random_scale_inds)

  # 
  # #######################################################################
  # 

  # 
  rois_per_image = cfg.TRAIN.BATCH_SIZE / num_images
  # fg_rois_per_image = rois_per_image - 1
  fg_rois_per_image = cfg.TRAIN.FG_NUMBER
  if fg_rois_per_image > rois_per_image - 1:
    fg_rois_per_image = rois_per_image - 1
  # fg_rois_per_image = min(rois_per_image - 1, fg_rois_per_image)

  # Now, build the region of interest and label blobs -- (batch_image_idx, x, y, w, h)
  rois_blob = np.zeros((0, 5), dtype=np.float32)
  labels_blob = np.zeros((0), dtype=np.float32)
  n_rois_count_blob = np.zeros((0), dtype=np.float32)
  # 
  bbox_targets_blob = np.zeros((0, 4 * num_classes), dtype=np.float32)
  bbox_loss_blob = np.zeros(bbox_targets_blob.shape, dtype=np.float32)

  # # Used for visualizations
  # has_visual_bbox = True
  if has_visual_bbox:
    all_overlaps = []

  # 
  # #######################################################################
  # 

  # Begin
  # hstack or vstack: stack the element by column or row
  for im_i in xrange(num_images):
    # 
    img_shape = im_shapes[im_i]
    im_rois, labels, overlaps, n_rois_count, bbox_targets, bbox_loss \
        = _sample_rois(roidb[im_i], img_shape, rois_per_image, fg_rois_per_image,
                       num_classes)

    # Add to RoIs blob
    rois = _project_im_rois(im_rois, im_scales[im_i])
    # (n, x, y, w, h) specifying an image batch index n and a
    batch_ind = im_i * np.ones((rois.shape[0], 1))

    rois_blob_this_image = np.hstack((batch_ind, rois))
    rois_blob = np.vstack((rois_blob, rois_blob_this_image))

    # Add to labels, bbox targets, and bbox loss blobs
    labels_blob = np.hstack((labels_blob, labels))
    n_rois_count_blob = np.hstack((n_rois_count_blob, n_rois_count))

    # 
    bbox_targets_blob = np.vstack((bbox_targets_blob, bbox_targets))
    bbox_loss_blob = np.vstack((bbox_loss_blob, bbox_loss))

    # # Used for visualizations
    if has_visual_bbox:
      all_overlaps = np.hstack((all_overlaps, overlaps))

  # # For debug visualizations
  if has_visual_bbox:
    image_list = [roidb[il]['image'] for il in xrange(len(roidb))]
    image_list = [il.split(os.sep)[-1] for il in image_list]
    _vis_minibatch(im_blob, rois_blob, labels_blob, all_overlaps, image_list)

  # 
  # #######################################################################
  # 
  blobs = {'data': im_blob,
           'rois': rois_blob,
           'labels': labels_blob,
           'n_rois_count': n_rois_count_blob
           }
  if cfg.TRAIN.BBOX_REG:
    blobs['bbox_targets'] = bbox_targets_blob
    blobs['bbox_loss_weights'] = bbox_loss_blob

  return blobs




# Visualize
def _vis_minibatch(im_blob, rois_blob, labels_blob, overlaps, image_list):
  """Visualize a mini-batch for debugging."""
  import matplotlib.pyplot as plt
  rb_len = rois_blob.shape[0]
  if cfg.TRAIN.VISUAL_BBOX_LEN > 0:
    rb_len = min(cfg.TRAIN.VISUAL_BBOX_LEN, rb_len)

  for i in xrange(rb_len):
    rois = rois_blob[i, :]
    # 
    im_ind = rois[0]
    roi = rois[1:]
    # 
    im = im_blob[im_ind, :, :, :].transpose((1, 2, 0)).copy()
    im += cfg.PIXEL_MEANS
    im = im[:, :, (2, 1, 0)]
    im = im.astype(np.uint8)
    # 
    cls = labels_blob[im_ind]
    plt.imshow(im)
    # 
    print 'im_ind:', im_ind, ', image path:', image_list[int(im_ind)], ', class:', cls, ', overlap: ', overlaps[i]
    
    plt.gca().add_patch(
        plt.Rectangle((roi[0], roi[1]), roi[2] - roi[0],
                      roi[3] - roi[1], fill=False,
                      edgecolor='r', linewidth=3)
        )
    # 
    plt.show()