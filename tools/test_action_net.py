#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Test a Fast R-CNN network on an image database."""

import _init_paths
# 
import caffe
# from fast_rcnn.test_action import test_net
from fast_rcnn.test_action import test_net
from fast_rcnn.config_action import cfg, global_dict, cfg_from_file, get_output_dir
from action_datasets.factory import get_imdb
import action_roi_data_layer.roidb as ardl_roidb
# 
import argparse
import pprint
import time, os, sys



def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
  # gpu
  parser.add_argument('--gpu', dest='gpu_id', help='GPU id to use',
                      default=0, type=int)
  # def
  parser.add_argument('--def', dest='prototxt',
                      help='prototxt file defining the network',
                      default=None, type=str)
  # net
  parser.add_argument('--net', dest='caffemodel',
                      help='model to test',
                      default=None, type=str)
  # the action label-anno file
  # annos
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
                      help='''dataset to train on, the format of `imdb_name` 
                              is `rcnn_<datatype>_<split>, 
                              where datatype is the name of action dataset, like:
                              PascalVoc2012, Willowactions, Stanford, MPII
                              where split indicates the train/val/trainval/test.''',
                      default='rcnn_PascalVoc2012_trainval', type=str)
  # wait
  parser.add_argument('--wait', dest='wait',
                      help='wait until net file exists',
                      default=True, type=bool)
  # comp
  parser.add_argument('--comp', dest='comp_mode', help='competition mode',
                      action='store_true')

  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)

  args = parser.parse_args()
  return args


def test_action_net():
  # Get args
  args = parse_args()
  print('Called with args:')
  print '*************************************************************'
  print(args)
  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
  print '*************************************************************'
  print('Using config:')
  pprint.pprint(cfg)
  print

  sleep_time = 10
  while not os.path.exists(args.caffemodel) and args.wait:
    print('Waiting for {} to exist...'.format(args.caffemodel))
    time.sleep(sleep_time)

  # Set GPU
  caffe.set_mode_gpu()
  caffe.set_device(args.gpu_id)
  # Initialize Net
  net = caffe.Net(args.prototxt, args.caffemodel, caffe.TEST)
  trained_modelname = os.path.splitext(os.path.basename(args.caffemodel))[0]

  print 
  print "imdb_name:", args.imdb_name
  print
  imdb = get_imdb(args.imdb_name)  
  imdb.competition_mode(args.comp_mode)
  # Get output directory
  sub_dir = "action_output"
  output_dir = get_output_dir(imdb=imdb, sub_dir=sub_dir)
  print 'Output will be saved to `{:s}`'.format(output_dir)
  # 
  exper_name = args.exper_name
  if exper_name:
    output_dir = os.path.join(output_dir, exper_name, trained_modelname)
    print
    print "exper_name:", exper_name
    print "output_dir(new):", output_dir
    print
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)


  # Process the roidb
  print 'Loaded dataset `{:s}` for val or test'.format(imdb.name)
  ardl_roidb.prepare_roidb(imdb)
  roidb = imdb.roidb
  # cfg.GLOBAL_DICTS[cfg.BBOX_MEANS], cfg.GLOBAL_DICTS[cfg.BBOX_STDS] = \
  #     ardl_roidb.add_bbox_regression_targets(roidb)
  bbox_means, bbox_stds = \
      ardl_roidb.add_bbox_regression_targets(roidb)

  # Get some info variables
  num_images = len(imdb.image_index)
  num_classes = imdb.num_classes
  # 
  # cfg.GLOBAL_DICTS[cfg.CLASS_TO_IND] = imdb.class_to_ind
  # cfg.GLOBAL_DICTS[cfg.IND_TO_CLASS] = imdb.ind_to_class
  class_to_ind = imdb.class_to_ind
  ind_to_class = imdb.ind_to_class

  global_dict[cfg.TEST_BBOX_MEANS] = bbox_means
  global_dict[cfg.TEST_BBOX_STDS] = bbox_stds
  global_dict[cfg.CLASS_TO_IND] = class_to_ind
  global_dict[cfg.IND_TO_CLASS] = ind_to_class
  
  # Test
  test_net(net, roidb, output_dir, num_images, num_classes)



if __name__ == '__main__':
  ''''''
  test_action_net()
