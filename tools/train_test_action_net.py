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
from fast_rcnn.train_test_action import train_test_action
# 
import pprint
import argparse
import numpy as np
import time, os, sys


def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train&Test a Fast R-CNN network')

  # gpu
  parser.add_argument('--gpu', dest='gpu_id', help='GPU id to use',
                      default=0, type=int)
  # solver -- train
  parser.add_argument('--solver', dest='solver',
                      help='solver prototxt',
                      default=None, type=str)
  # weights -- train
  parser.add_argument('--weights', dest='pretrained_model',
                      help='initialize with pretrained model weights',
                      default=None, type=str)
  # def -- test
  parser.add_argument('--def', dest='prototxt',
                      help='prototxt file defining the network',
                      default=None, type=str)
  # net -- test
  parser.add_argument('--net', dest='caffemodel',
                      help='model to test',
                      default=None, type=str)
  # iters
  parser.add_argument('--iters', dest='max_iters',
                      help='number of iterations to train',
                      default=160000, type=int)
  # re_iter
  parser.add_argument('--re_iter', dest='re_iter',
                      help='the iterations of pretrained model, when retrain the network',
                      default=None, type=str)

  # exper_name
  parser.add_argument('--exper_name', dest='exper_name',
                      help='to mark the different experiment names',
                      default=None, type=str)
  # imdb
  parser.add_argument('--train_imdb', dest='train_imdb_name',
                      help='''dataset to train on, the format of `imdb_name` is `rcnn_<datatype>_<split>, 
                              where datatype is the name of action dataset, like:
                              PascalVoc2012, Willowactions, Stanford, MPII
                              where split indicates the train/val/trainval/test.''',
                      default='rcnn_PascalVoc2012_train', type=str)
  # imdb
  parser.add_argument('--test_imdb', dest='test_imdb_name',
                      help='''dataset to train on, the format of `imdb_name` 
                              is `rcnn_<datatype>_<split>, 
                              where datatype is the name of action dataset, like:
                              PascalVoc2012, Willowactions, Stanford, MPII
                              where split indicates the train/val/trainval/test.''',
                      default='rcnn_PascalVoc2012_val', type=str)
  

  # the action label-anno file
  # annos
  parser.add_argument('--annos', dest='annosfile',
                      help='used for getting the ground truth label, like img_idx, bbox, action_label',
                      default=None, type=str)
  # cfg
  parser.add_argument('--cfg', dest='cfg_file',
                      help='optional config file',
                      default=None, type=str)
  # wait
  parser.add_argument('--wait', dest='wait',
                      help='wait until net file exists',
                      default=True, type=bool)
  # comp
  parser.add_argument('--comp', dest='comp_mode', help='competition mode',
                      action='store_true')
  # rand
  parser.add_argument('--rand', dest='randomize',
                      help='randomize (do not use a fixed seed)',
                      action='store_true')

  print 
  print "Parsing args Done!"
  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)

  args = parser.parse_args()
  return args


# #########################################################################
# 
# 
# #########################################################################


def train_test_action_net():
  args = parse_args()
  train_test_action(args)

  

if __name__ == '__main__':
  '''
  Train and Test jointly happens
  '''
  train_test_action_net()

