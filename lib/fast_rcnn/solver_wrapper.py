# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# 
# Modified by ddk
# --------------------------------------------------------

"""Train&TEST a Fast R-CNN network..."""

import caffe
import os, sys
import numpy as np
from utils.timer import Timer
import google.protobuf as pb2
from caffe.proto import caffe_pb2
from fast_rcnn.config_action import cfg, global_dict, get_output_dir2
from fast_rcnn.test_action import test_net


# 
class SolverWrapper(object):
  """A simple wrapper around Caffe's solver.
  This wrapper gives us control over he snapshotting process, which we
  use to unnormalize the learned bounding-box regression weights.
  """

  def __init__(self, solver_prototxt, train_roidb, output_dir, test_net, test_roidb, \
      pretrained_model=None, re_iter=None, max_iters=16000):
    """Initialize the SolverWrapper."""
    self.output_dir = output_dir
    self.re_iter = re_iter
    self.max_iters = max_iters
    
    # Initialize Solver
    print "Using SGDSolver for network..."
    self.solver = caffe.SGDSolver(solver_prototxt)

    print "Get solver_param..."
    if pretrained_model:
      print ('Loading pretrained model '
             'weights from {:s}').format(pretrained_model)
      self.solver.net.copy_from(pretrained_model)
    self.solver_param = caffe_pb2.SolverParameter()
    # 
    with open(solver_prototxt, 'rt') as f:
      pb2.text_format.Merge(f.read(), self.solver_param)

    # Set train_roidb for the datalayer in `../action_roi_data_layer/layer.py - set_roidb`
    self.solver.net.layers[0].set_roidb(train_roidb)

    # 
    self.test_net = test_net
    self.test_roidb = test_roidb
    self.test_interval = cfg.TEST.TEST_INTERVAL
    self.start_test_iter = cfg.TEST.START_TEST_ITER

  
  def snapshot(self):
    """Take a snapshot of the network after unnormalizing the learned
    bounding-box regression weights. This enables easy use at test-time.
    """
    net = self.solver.net

    if cfg.TRAIN.BBOX_REG:
      # save original values
      orig_0 = net.params['bbox_pred'][0].data.copy()
      orig_1 = net.params['bbox_pred'][1].data.copy()

      # scale and shift with bbox reg unnormalization; then save snapshot
      net.params['bbox_pred'][0].data[...] = \
              (net.params['bbox_pred'][0].data *
               self.bbox_stds[:, np.newaxis])
      net.params['bbox_pred'][1].data[...] = \
              (net.params['bbox_pred'][1].data *
               self.bbox_stds + self.bbox_means)
              
    #  build the directory
    if not os.path.exists(self.output_dir):
      os.makedirs(self.output_dir)

    infix = ('_' + cfg.TRAIN.SNAPSHOT_INFIX
             if cfg.TRAIN.SNAPSHOT_INFIX != '' else '')
    filename = (self.solver_param.snapshot_prefix + infix +
                '_iter_{:d}'.format(self.solver.iter) + '.caffemodel')
    filename = os.path.join(self.output_dir, filename)

    print "snapshoted filename:", filename
    net.save(str(filename))
    print 'Wrote snapshot to: {:s}'.format(filename)

    if cfg.TRAIN.BBOX_REG:
      # restore net to original state
      net.params['bbox_pred'][0].data[...] = orig_0
      net.params['bbox_pred'][1].data[...] = orig_1


  def _get_test_output_dir(self, caffe_model_name, imdb_name, exper_name, sub_dir="action_output"):
    # 
    model_name = os.path.splitext(os.path.basename(caffe_model_name))[0]

    # 
    output_dir = get_output_dir2(imdb_name, sub_dir=sub_dir)
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

    print 'Output int the testing stage will be saved to `{:s}`'.format(output_dir)
    global_dict[cfg.TEST_OUTPUT_DIR] = output_dir

    if not os.path.exists(output_dir):
      os.makedirs(output_dir)
    return output_dir


  def test_model(self):
    # 
    self.test_net.share_with(self.solver.net)

    # 
    infix = ('_' + cfg.TRAIN.SNAPSHOT_INFIX
             if cfg.TRAIN.SNAPSHOT_INFIX != '' else '')
    caffe_model_name = (self.solver_param.snapshot_prefix + infix +
                '_iter_{:d}'.format(self.solver.iter))
    exper_name = global_dict[cfg.EXPERIMENT_NAME]
    imdb_name = global_dict[cfg.TEST_IMDB_NAME]
    # 
    output_dir = self._get_test_output_dir(caffe_model_name, imdb_name, exper_name)

    # 
    num_images = global_dict[cfg.TEST_NUM_IMAGES]
    num_classes = global_dict[cfg.NUM_CLASSES]
    # 
    test_net(self.test_net, self.test_roidb, output_dir, num_images, num_classes)


  def train_model(self):
    """Network training loop."""
    last_snapshot_iter = -1
    if self.re_iter is not None:
      re_iter = int(self.re_iter)
      if re_iter > 0:
        print
        print "retrain network from", re_iter, "iteration!"
        print
        import time
        time.sleep(2)
        self.solver.set_iter(re_iter)
    # 
    timer = Timer()
    print
    print
    print "start train network!"

    while self.solver.iter < self.max_iters:
      # Timer starts
      timer.tic()

      # Make one SGD update -- Forward & Backward
      self.solver.step(1)

      # Timer ends
      timer.toc()

      # Display loss 
      if self.solver.iter % (10 * self.solver_param.display) == 0:
        print 'speed: {:.3f}s / iter'.format(timer.average_time)

      # Snapshot
      if self.solver.iter % cfg.TRAIN.SNAPSHOT_ITERS == 0:
        last_snapshot_iter = self.solver.iter
        self.snapshot()

      # Test
      if self.solver.iter >= self.start_test_iter and \
          self.solver.iter % self.test_interval == 0:
        self.test_model()

    # End training and snapshot the last iteration model
    if last_snapshot_iter != self.solver.iter:
      self.snapshot()

    print
    print "Training network has been done!"
    print