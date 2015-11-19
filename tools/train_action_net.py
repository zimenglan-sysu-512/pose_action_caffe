#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# 
# Modified by ddk
# --------------------------------------------------------

"""Train a Fast R-CNN network on a region of interest database."""
import _init_paths
import caffe
from fast_rcnn.train_action  import get_training_roidb, train_net
from fast_rcnn.config_action import cfg, cfg_from_file, get_output_dir
from action_datasets.factory import get_imdb

import time
import pprint
import os, sys
import argparse
import numpy as np


# 
def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
  # gpu
  parser.add_argument('--gpu', dest='gpu_id',
                      help='GPU device id to use [0]',
                      default=0, type=int)
  # solver
  parser.add_argument('--solver', dest='solver',
                      help='solver prototxt',
                      default=None, type=str)
  # iters
  parser.add_argument('--iters', dest='max_iters',
                      help='number of iterations to train',
                      default=480000, type=int)
  # weights
  parser.add_argument('--weights', dest='pretrained_model',
                      help='initialize with pretrained model weights',
                      default=None, type=str)
  # the action label-anno file
  parser.add_argument('--annos', dest='annosfile',
                      help='used for getting the ground truth label, like img_idx, bbox, action_label',
                      default=None, type=str)
  # exper_name
  parser.add_argument('--exper_name', dest='exper_name',
                      help='to mark the different experiment names',
                      default=None, type=str)
  # re_iter
  parser.add_argument('--re_iter', dest='re_iter',
                      help='the iterations of pretrained model, when retrain the network',
                      default=None, type=str)
  # cfg
  parser.add_argument('--cfg', dest='cfg_file',
                      help='optional config file',
                      default=None, type=str)
  # imdb
  parser.add_argument('--imdb', dest='imdb_name',
                      help='''dataset to train on, the format of `imdb_name` is `rcnn_<datatype>_<split>, 
                              where datatype is the name of action dataset, like:
                              PascalVoc2012, Willowactions, Stanford, MPII
                              where split indicates the train/val/trainval/test.''',
                      default='rcnn_PascalVoc2012_train', type=str)
  # wait
  parser.add_argument('--wait', dest='wait',
                      help='wait until net file exists',
                      default=True, type=bool)
  # rand
  parser.add_argument('--rand', dest='randomize',
                      help='randomize (do not use a fixed seed)',
                      action='store_true')

  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)

  args = parser.parse_args()
  return args


# Train a rcnn network for acton recognition in still/static images
def train_action_net():
  args = parse_args()

  print('Called with args:')
  print(args)
  print
  print

  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)

  print('Using config:')
  pprint.pprint(cfg)
  print
  print

  # Fix the random seeds (numpy and caffe) for reproducibility
  if not args.randomize:
    np.random.seed(cfg.RNG_SEED)
    caffe.set_random_seed(cfg.RNG_SEED)

  # Set up caffe
  caffe.set_mode_gpu()
  if args.gpu_id is not None:
    caffe.set_device(args.gpu_id)

  # 
  print "args.imdb_name:", args.imdb_name

  # get_imdb(args.imdb_name) == action_datasets.rcnn_action(datatype, split)
  #   where datatype is the name of action dataset, like:
  #     PascalVoc2012, Willowactions, Stanford, MPII
  #   where split indicates the train/val/trainval/test
  # instance of action_datasets.rcnn_action, **note that rcnn_action inherits imdb**
  # 
  # imdb_name: rcnn_<datatype>_<imageset>
  #   where datatype: PascalVoc2012, Willowactions, Stanford, MPII, or more...
  #   imageset: train, val, trainval, test
  imdb = get_imdb(args.imdb_name)

  # 
  print 'Loaded dataset `{:s}` for training'.format(imdb.name)
  roidb = get_training_roidb(imdb)

  # 
  sub_dir = "action_output"
  output_dir = get_output_dir(imdb=imdb, sub_dir=sub_dir, net=None)
  print 'Output will be saved to `{:s}`'.format(output_dir)

  exper_name = args.exper_name
  if exper_name:
    print "exper_name:", exper_name
  re_iter = args.re_iter

  sleep_time = 5
  while not os.path.exists(args.pretrained_model) and args.wait:
    print('Waiting for {} to exist...'.format(args.pretrained_model))
    time.sleep(sleep_time)
  print "Initialize from", args.pretrained_model
  # 
  train_net(args.solver, roidb, output_dir,
            pretrained_model=args.pretrained_model, \
            max_iters=args.max_iters, \
            exper_name=exper_name, \
            re_iter=re_iter)


# 
if __name__ == '__main__':
  '''
  Train a fast-rcnn net for action recognition in still/static images.
  '''
  train_action_net()   