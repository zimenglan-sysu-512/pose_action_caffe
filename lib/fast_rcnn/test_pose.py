# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Test a Fast R-CNN network on an imdb (image database)."""

from fast_rcnn.config_pose import pose_cfg
import caffe
from utils.timer import Timer
from utils.blob import im_list_to_blob
import cv2
import os, sys
import cPickle
import argparse
import numpy as np

def _create_dire(path):
  if not os.path.exists(path):
    os.makedirs(path)

def _im_list2blob(ims, is_color=True):
    """
    Convert a list of images into a network input.
    Assumes images are already prepared (means subtracted, BGR order, ...).
    """
    num_images = len(ims)
    max_shape  = np.array([im.shape for im in ims]).max(axis=0)
    if is_color:
      blob     = np.zeros((num_images, max_shape[0], max_shape[1], 3), dtype=np.float32)
    else:
      blob     = np.zeros((num_images, max_shape[0], max_shape[1], 1), dtype=np.float32)
    
    for i in xrange(num_images):
      im = ims[i]
      if is_color:
        blob[i, 0:im.shape[0], 0:im.shape[1], :] = im
      else:
        blob[i, 0:im.shape[0], 0:im.shape[1], 0] = im
    
    # Move channels (axis 3) to axis 1
    # Axis order will become: (batch elem, channel, height, width)
    channel_swap = (0, 3, 1, 2)
    blob = blob.transpose(channel_swap)
    return blob

def _prep_im4blob(im, min_size, max_size, pixel_means=None):
    """Mean subtract and scale an image for use in a blob."""
    # cv2.imshow("1", im)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    im = im.astype(np.float32, copy=False)

    # print im
    # cv2.imshow("2", im)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    if pixel_means is not None:
      print "subtract pixel means"
      im -= pixel_means

    im_shape    = im.shape
    im_min_size = np.min(im_shape[0:2])
    im_max_size = np.max(im_shape[0:2])
    im_scale    = float(min_size) / float(im_min_size)

    if np.round(im_scale * im_max_size) > max_size:
      im_scale  = float(max_size) / float(im_max_size)

    im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale, \
                    interpolation=cv2.INTER_LINEAR)

    return im, im_scale

def _check_box(x1, y1, x2, y2, w, h):
  assert x1 >= 0
  assert y1 >= 0
  assert x2 <= w - 1
  assert y2 <= h - 1
  assert x2 >= x1
  assert y2 >= y1

def _get_blobs(im_path, pt_boxes):
  assert len(pt_boxes) >= 1
  if not os.path.exists(im_path):
    raise IOError(('{:s} not found.\n').format(im_path))

  im = cv2.imread(im_path)
  h, w, _ = im.shape
  
  crop_ims    = []
  torso_masks = []
  im_scales   = []
  min_size    = pose_cfg.TEST.MIN_SIZE
  max_size    = pose_cfg.TEST.MAX_SIZE

  im_c        = 0
  aux_infos   = []
  n_pt_box    = len(pt_boxes)

  for pt_box in pt_boxes:
    assert len(pt_box) == 2
    p_box, t_box = pt_box
    assert len(p_box) == 4
    assert len(t_box) == 4
    p_x1, p_y1, p_x2, p_y2 = p_box
    p_x1 = max(p_x1 - pose_cfg.TEST.P_DXY, 1)
    p_xy = max(p_y1 - pose_cfg.TEST.P_DXY, 1)
    p_x2 = min(p_x2 + pose_cfg.TEST.P_DXY, w - 1)
    p_y2 = min(p_y2 + pose_cfg.TEST.P_DXY, h - 1)
    _check_box(p_x1, p_y1, p_x2, p_y2, w,  h)

    im2 = im[p_y1: p_y2+1, p_x1: p_x2+1] # crop
    h2, w2, _ = im2.shape

    t_x1, t_y1, t_x2, t_y2 = t_box
    t_x1 = t_x1 - p_x1
    t_y1 = t_y1 - p_y1
    t_x2 = t_x2 - p_x1
    t_y2 = t_y2 - p_y1
    _check_box(t_x1, t_y1, t_x2, t_y2, w2, h2)
    if t_x1 > t_x2:
      t_x1, t_x2 = t_x2, t_x1
    if t_y1 > t_y2:
      t_y1, t_y2 = t_y2, t_y1
    tw   = t_x2 - t_x1
    if pose_cfg.TEST.TW_RATIO > 0:
      t_x1 = t_x1 + tw * pose_cfg.TEST.TW_RATIO
    elif pose_cfg.TEST.TW_RATIO < 0:
      t_x2 = t_x2 + tw * pose_cfg.TEST.TW_RATIO
    _check_box(t_x1, t_y1, t_x2, t_y2, w2, h2)

    # resize 
    im2, im_scale = _prep_im4blob(im2, min_size, max_size)
    h22, w22, _     = im2.shape
    t_x1 = t_x1 * im_scale
    t_y1 = t_y1 * im_scale
    t_x2 = t_x2 * im_scale
    t_y2 = t_y2 * im_scale
    t_x1 = int(t_x1)
    t_y1 = int(t_y1)
    t_x2 = int(t_x2)
    t_y2 = int(t_y2)
    im3 = np.zeros((h22, w22), dtype=np.uint8)
    im3[t_y1: t_y2+1, t_x1: t_x2+1] = pose_cfg.TEST.FILL_VAL

    crop_ims.append(im2)
    torso_masks.append(im3)
    im_scales.append(im_scale)

    # imgidx, width, height, scale, flippable
    aux_info = [im_c, w2, h2, im_scale, 0]
    aux_infos.append(aux_info)

    im_c     = im_c + 1

  # if pose_cfg.TEST.SHOW_IMGS:
  #   im_name  = os.path.basename(im_path)
  #   im_name2 = im_name.rsplit(".", 1)[0]
  #   for c in xrange(im_c):
  #     im_name3 = im_name2 + "_" + str(c)
  #     im2      = crop_ims[c]
  #     cv2.imshow(im_name3, im2)
  #     cv2.waitKey(0)
  #     cv2.destroyAllWindows()
  #     im3      = torso_masks[c]
  #     cv2.imshow(im_name3, im3)
  #     cv2.waitKey(0)
  #     cv2.destroyAllWindows()

  crop_ims_blob    = _im_list2blob(crop_ims,    is_color=True)
  torso_masks_blob = _im_list2blob(torso_masks, is_color=False)

  assert n_pt_box == im_c
  assert n_pt_box == len(aux_infos)
  aux_infos_blob   = np.zeros((n_pt_box, 5), dtype=np.float32)
  for c in xrange(im_c):
    aux_infos_blob[c, :] = aux_infos[c]

  return crop_ims_blob, torso_masks_blob, aux_infos_blob
  
def _viz_inds():
  parts_num  = pose_cfg.TEST.PARTS_NUM
  parts_inds = pose_cfg.TEST.DRAW_PARTS_INDS
  skels_inds = pose_cfg.TEST.DRAW_SKELS_INDS

  parts_inds = parts_inds.strip()
  parts_inds = parts_inds.split(pose_cfg.COMMA)
  parts_inds = [int(pi.strip()) for pi in parts_inds]
  assert len(parts_inds) <= parts_num
  assert len(parts_inds) >= 0
  for pi in parts_inds:
    assert pi >= 0
    assert pi <= parts_num

  skels_inds = skels_inds.strip()
  skels_inds = skels_inds.split(pose_cfg.COMMA)
  points     = []
  for si in skels_inds:
    si = si.strip().split(pose_cfg.COLON)
    si = [int(i.strip()) for i in si]
    assert len(si) == 2
    assert si[0] in parts_inds
    assert si[1] in parts_inds
    points.append((si[0], si[1]))
  skels_inds = points

  return parts_inds, skels_inds

def _viz_pose(im_path, pred_coords, pt_boxes):
  im       = cv2.imread(im_path)
  parts_inds, skels_inds = _viz_inds()
  
  radius      = pose_cfg.RADIUS
  l_thickness = pose_cfg.L_THICKNESS
  r_thickness = pose_cfg.R_THICKNESS
  colors      = pose_cfg.VIZ_COLORS
  n_colors    = len(colors) - 6

  viz_dire    = pose_cfg.TEST.VIZ_DIRE
  _create_dire(viz_dire)

  im_name  = os.path.basename(im_path)
  out_path = viz_dire + im_name

  nc, cc, hc, wc = pred_coords.shape
  n_pt_box       = len(pt_boxes)
  assert nc == n_pt_box

  parts_num = pose_cfg.TEST.PARTS_NUM

  for n in xrange(nc):
    coords = pred_coords[n, :, 0, 0]
    assert len(coords) == parts_num * 2

    pt_box = pt_boxes[n]
    p_box, t_box = pt_box
    p_x1, p_y1, p_x2, p_y2 = p_box
    t_x1, t_y1, t_x2, t_y2 = t_box
    
    # back to origin image axis
    for j in xrange(parts_num):
      j2 = j * 2
      coords[j2 + 0] += p_x1
      coords[j2 + 1] += p_y1

    # joint
    for c in xrange(len(parts_inds)):
      j1 = parts_inds[c]
      j2 = j1 * 2
      x  = coords[j2 + 0] 
      y  = coords[j2 + 1]
      p  = (x, y)
      cv2.circle(im, p, radius, colors[c % n_colors], r_thickness)

    # line between joints
    for pi in skels_inds:
      j1 = pi[0]
      j1 = j1 * 2
      x  = coords[j1 + 0]
      y  = coords[j1 + 1]
      p1 = (x , y)
      j2 = pi[1]
      j2 = j2 * 2
      x  = coords[j2 + 0]
      y  = coords[j2 + 1]
      p2 = (x , y)
      cv2.line(im, p1, p2, colors[n_colors + 3], l_thickness)

    # person and torso bounding box
    p1 = (p_x1, p_y1)
    p2 = (p_x2, p_y2)
    cv2.rectangle(im, p1, p2, colors[n_colors + 1], 3)
    p1 = (t_x1, t_y1)
    p2 = (t_x2, t_y2)
    cv2.rectangle(im, p1, p2, colors[n_colors + 2], 3)

  cv2.imwrite(out_path, im)

  return out_path
  
def pose_estimation(net, im_path, pt_boxes):
  ''''''
  data_name     = pose_cfg.TEST.DATA_LAYER_NAME
  tmasks_name   = pose_cfg.TEST.TORSO_MASK_LAYER_NAME
  aux_info_name = pose_cfg.TEST.AUX_INFO_LAYER_NAME
  target_name   = pose_cfg.TEST.TARGET_LAYER_NAME

  blobs = {data_name : None, tmasks_name : None, aux_info_name: None}
  crop_ims_blob, torso_masks_blob, aux_infos_blob = _get_blobs(im_path, pt_boxes)
  blobs[data_name]     = crop_ims_blob
  blobs[tmasks_name]   = torso_masks_blob
  blobs[aux_info_name] = aux_infos_blob

  out_names_blobs = [target_name,]

  # reshape network inputs
  net.blobs[data_name].reshape(*(blobs[data_name].shape))
  net.blobs[tmasks_name].reshape(*(blobs[tmasks_name].shape))
  net.blobs[aux_info_name].reshape(*(blobs[aux_info_name].shape))

  kwargs = {}
  kwargs[data_name]     = blobs[    data_name].astype(np.float32, copy=False)
  kwargs[tmasks_name]   = blobs[  tmasks_name].astype(np.float32, copy=False)
  kwargs[aux_info_name] = blobs[aux_info_name].astype(np.float32, copy=False)
  # forward
  blobs_out = net.forward(blobs=out_names_blobs, start=None, end=None, **kwargs)

  # get estimation coordinates
  parts_num   = pose_cfg.TEST.PARTS_NUM
  pred_coords = blobs_out[target_name]    # has been rescaled
  # print "pred_coords.shape", pred_coords.shape # (1, 28, 1, 1)

  out_path = _viz_pose(im_path, pred_coords, pt_boxes)
  return out_path
  

if __name__ == '__main__':
  ''''''
  pass