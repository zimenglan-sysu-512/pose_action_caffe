#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Train&Test a Fast R-CNN network on an image database."""

import _init_paths
# 
import caffe
from action_datasets.factory import get_imdb
import action_roi_data_layer.roidb as ardl_roidb
from fast_rcnn.solver_wrapper import SolverWrapper
from fast_rcnn.config_action import cfg, global_dict, cfg_from_file, get_output_dir2
# 
import pprint
import argparse
import numpy as np
import time, os, sys


def _init_config(args):
  print
  print('Called with args:')
  print(args)
  print
  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
  print('Using config:')
  pprint.pprint(cfg)
  print
  print
  return args


def _set_caffe_settings(gpu_id=0, randomize=False):
  # Set GPU
  caffe.set_mode_gpu()
  caffe.set_device(gpu_id)
  # Fix the random seeds (numpy and caffe) for reproducibility
  if not randomize:
    np.random.seed(cfg.RNG_SEED)
    caffe.set_random_seed(cfg.RNG_SEED)


def _get_caffemodel_name(caffe_model_name):
  model_name = os.path.splitext(os.path.basename(caffe_model_name))[0]
  return model_name


def _get_output_dir(imdb_name, net=None, sub_dir="", exper_name="", model_name=""):
  output_dir = get_output_dir2(imdb_name, sub_dir=sub_dir, net=net)
  if exper_name:
    print "exper_name:", exper_name
    output_dir = os.path.join(output_dir, exper_name)
  if model_name:
    print "model_name:", model_name
    output_dir = os.path.join(output_dir, model_name)
  # 
  print 'Output will be saved to `{:s}`'.format(output_dir)
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  return output_dir


def _get_test_output_dir(caffe_model_name, imdb_name, exper_name, sub_dir="action_output"):
  ''''''
  model_name = _get_caffemodel_name(caffe_model_name)
  output_dir = _get_output_dir(imdb_name, sub_dir=sub_dir, \
    model_name=model_name, exper_name=exper_name)
  print 'Output int the testing stage will be saved to `{:s}`'.format(phase, output_dir)
  global_dict[cfg.TEST_OUTPUT_DIR] = output_dir

  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  return output_dir


# #########################################################################
# 
# 
# #########################################################################


def _create_train_roidb(imdb_name, exper_name, sub_dir = "action_output"):
  ''''''
  print "imdb_name in trining stage is `{:s}`".format(imdb_name)
  imdb = get_imdb(imdb_name)
  print 'Loaded dataset `{:s}` for training'.format(imdb.name)
  # 
  if cfg.TRAIN.USE_FLIPPED:
    print 'Appending horizontally-flipped training examples...'
    imdb.append_flipped_images()
    print 'Horizontally-flipping done!'
    print
  # 
  print 'Preparing training data...'
  ardl_roidb.prepare_roidb(imdb, cfg.TRAIN_PHASE)
  print 'Preparing training data done!'
  print
  roidb = imdb.roidb

  # 
  print 
  print 'Computing bounding-box regression targets...'
  bbox_means, bbox_stds = \
          ardl_roidb.add_bbox_regression_targets(roidb)
  print 'Computing bounding-box regression targets done...'
  print
  # 
  output_dir = _get_output_dir(imdb_name, sub_dir=sub_dir, exper_name=exper_name)
  print 'Output int testing stage will be saved to `{:s}`'.format(output_dir)

  # get some info variables
  num_images = len(imdb.image_index)
  num_classes = imdb.num_classes
  class_to_ind = imdb.class_to_ind
  ind_to_class = imdb.ind_to_class
  # 
  if cfg.TRAIN_BBOX_MEANS not in global_dict.keys():
   global_dict[cfg.TRAIN_BBOX_MEANS] = bbox_means

  if cfg.TRAIN_BBOX_STDS not in global_dict.keys():
   global_dict[cfg.TRAIN_BBOX_STDS] = bbox_stds

  if cfg.CLASS_TO_IND not in global_dict.keys():
   global_dict[cfg.CLASS_TO_IND] = class_to_ind

  if cfg.IND_TO_CLASS not in global_dict.keys():
   global_dict[cfg.IND_TO_CLASS] = ind_to_class

  if cfg.NUM_CLASSES not in global_dict.keys():
   global_dict[cfg.NUM_CLASSES] = num_classes
   
  if cfg.TRAIN_NUM_IMAGES not in global_dict.keys():
   global_dict[cfg.TRAIN_NUM_IMAGES] = num_images

  if cfg.TRAIN_OUTPUT_DIR not in global_dict.keys():
    global_dict[cfg.TRAIN_OUTPUT_DIR] = output_dir

  if cfg.TRAIN_IMDB not in global_dict.keys():
    global_dict[cfg.TRAIN_IMDB] = imdb

  if cfg.TRAIN_ROIDB not in global_dict.keys():
    global_dict[cfg.TRAIN_ROIDB] = roidb

  if cfg.TRAIN_IMDB_NAME not in global_dict.keys():
    global_dict[cfg.TRAIN_IMDB_NAME] = imdb_name


def _creat_slover_wrapper(solver_prototxt, train_roidb, output_dir, \
    test_net, test_roidb, pretrained_model, re_iter, max_iters, has_wait=False):
  ''''''
  sleep_time = 5
  while not os.path.exists(pretrained_model) and has_wait:
    print('Waiting for {} to exist...'.format(pretrained_model))
    time.sleep(sleep_time)
  print "Initialize from", pretrained_model

  solver_wrapper = SolverWrapper(solver_prototxt, \
      train_roidb, output_dir, test_net, test_roidb, \
      pretrained_model, re_iter, max_iters)

  return solver_wrapper


def _create_test_roidb(imdb_name, comp_mode, phase="val"):
  print 
  print "imdb_name in `{:s}` stage is `{:s}`".format(imdb_name, phase)
  # get imdb
  imdb = get_imdb(imdb_name)
  if comp_mode:
    imdb.competition_mode(comp_mode)
  print 'Loaded dataset `{:s}` in the {:s} stage'.format(imdb.name, phase)
  # 
  # get roidb
  print 'Preparing training data...'
  ardl_roidb.prepare_roidb(imdb)
  print 'Preparing training data done!'
  print
  roidb = imdb.roidb

  # Computing bounding-box regression
  print 
  print 'Computing bounding-box regression targets...'
  bbox_means, bbox_stds = \
          ardl_roidb.add_bbox_regression_targets(roidb)
  print 'Computing bounding-box regression targets done...'
  print

  # get some info variables
  num_images = len(imdb.image_index)
  num_classes = imdb.num_classes
  class_to_ind = imdb.class_to_ind
  ind_to_class = imdb.ind_to_class

  # 
  if cfg.TEST_BBOX_MEANS not in global_dict.keys():
    global_dict[cfg.TEST_BBOX_MEANS] = bbox_means

  if cfg.TEST_BBOX_STDS not in global_dict.keys():
    global_dict[cfg.TEST_BBOX_STDS] = bbox_stds

  if cfg.CLASS_TO_IND not in global_dict.keys():
    global_dict[cfg.CLASS_TO_IND] = class_to_ind
  if cfg.IND_TO_CLASS not in global_dict.keys():
    global_dict[cfg.IND_TO_CLASS] = ind_to_class

  if cfg.NUM_CLASSES not in global_dict.keys():
    global_dict[cfg.NUM_CLASSES] = num_classes

  if cfg.TEST_NUM_IMAGES not in global_dict.keys():
    global_dict[cfg.TEST_NUM_IMAGES] = num_images

  if cfg.TEST_IMDB not in global_dict.keys():
    global_dict[cfg.TEST_IMDB] = imdb

  if cfg.TEST_ROIDB not in global_dict.keys():
    global_dict[cfg.TEST_ROIDB] = roidb

  if cfg.TEST_IMDB_NAME not in global_dict.keys():
    global_dict[cfg.TEST_IMDB_NAME] = imdb_name
 

# prototxt is the test.prototxt
def _create_test_net(prototxt, caffemodel=None, has_wait=False):
  sleep_time = 10
  while not os.path.exists(prototxt) and has_wait:
    print('Waiting for {} to exist...'.format(prototxt))
    time.sleep(sleep_time)

  test_net = None
  if caffemodel:
    while not os.path.exists(caffemodel) and has_wait:
      print('Waiting for {} to exist...'.format(caffemodel))
      time.sleep(sleep_time)
    # Initialize Net
    test_net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    return test_net
  else:
    # Initialize Net
    test_net = caffe.Net(prototxt, caffe.TEST)

  return test_net



# #########################################################################
# 
# 
# #########################################################################


def train_test_action(args):
  args = _init_config(args)

  # Common
  has_wait = args.wait
  gpu_id = args.gpu_id
  randomize = args.randomize
  exper_name = args.exper_name

  # For training
  re_iter = args.re_iter
  max_iters = args.max_iters
  solver_prototxt = args.solver
  train_imdb_name = args.train_imdb_name
  pretrained_model = args.pretrained_model

  # For tesing
  prototxt = args.prototxt
  comp_mode = args.comp_mode
  caffemodel = args.caffemodel
  test_imdb_name = args.test_imdb_name

  # #####################################################
  # 
  # 
  # #####################################################

  if cfg.EXPERIMENT_NAME not in global_dict.keys():
    global_dict[cfg.EXPERIMENT_NAME] = exper_name

  # #####################################################
  # 
  # 
  # #####################################################

  # Set the caffe configuration
  _set_caffe_settings(gpu_id, randomize)
  
  # Create training and testing roidb
  _create_train_roidb(train_imdb_name, exper_name)
  # 
  _create_test_roidb(test_imdb_name, comp_mode)

  # 
  train_roidb = global_dict[cfg.TRAIN_ROIDB]
  train_output_dir = global_dict[cfg.TRAIN_OUTPUT_DIR]

  # 
  test_roidb = global_dict[cfg.TEST_ROIDB]
  test_net = _create_test_net(prototxt, has_wait=has_wait)

  # 
  solver_wrapper = _creat_slover_wrapper(solver_prototxt, \
      train_roidb, train_output_dir, test_net, test_roidb, \
      pretrained_model, re_iter, max_iters, has_wait)

  # train
  solver_wrapper.train_model()
