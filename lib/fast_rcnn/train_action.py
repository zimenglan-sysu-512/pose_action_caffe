# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# 
# Modified by ddk
# --------------------------------------------------------

"""Train a Fast R-CNN network."""

import caffe
from caffe.proto import caffe_pb2

# for action recognition task in still/static images
from fast_rcnn.config_action import cfg
import action_roi_data_layer.roidb as ardl_roidb

from utils.timer import Timer
import numpy as np
import os, sys
import google.protobuf as pb2



# 
class SolverWrapper(object):
  """A simple wrapper around Caffe's solver.
  This wrapper gives us control over he snapshotting process, which we
  use to unnormalize the learned bounding-box regression weights.
  """

  def __init__(self, solver_prototxt, roidb, output_dir,
               pretrained_model=None, re_iter=None):
    """Initialize the SolverWrapper."""
    self.output_dir = output_dir
    self.re_iter = re_iter

    print 'Computing bounding-box regression targets...'
    self.bbox_means, self.bbox_stds = \
            ardl_roidb.add_bbox_regression_targets(roidb)
    print 'Computing bounding-box regression targets done...'
    print

    
    # Initialize Solver
    print "Using SGDSolver for network..."
    self.solver = caffe.SGDSolver(solver_prototxt)

    print "Get solver_param..."
    if pretrained_model is not None:
      print ('Loading pretrained model '
             'weights from {:s}').format(pretrained_model)
      self.solver.net.copy_from(pretrained_model)
    self.solver_param = caffe_pb2.SolverParameter()
    # 
    with open(solver_prototxt, 'rt') as f:
      pb2.text_format.Merge(f.read(), self.solver_param)

    # Set roidb for the datalayer 
    # See ../action_roi_data_layer/layer.py - set_roidb
    self.solver.net.layers[0].set_roidb(roidb)

  
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


  def train_model(self, max_iters):
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

    while self.solver.iter < max_iters:
      timer.tic()
      # Make one SGD update -- forward & backward
      self.solver.step(1)
      timer.toc()
      # 
      if self.solver.iter % (10 * self.solver_param.display) == 0:
        print 'speed: {:.3f}s / iter'.format(timer.average_time)

      if self.solver.iter % cfg.TRAIN.SNAPSHOT_ITERS == 0:
        last_snapshot_iter = self.solver.iter
        self.snapshot()

    if last_snapshot_iter != self.solver.iter:
      self.snapshot()

    print
    print "Training network has been done!"
    print


# ##################################################


# 
def get_training_roidb(imdb):
  """Returns a roidb (Region of Interest database) for use in training."""
  if cfg.TRAIN.USE_FLIPPED:
    print 'Appending horizontally-flipped training examples...'
    imdb.append_flipped_images()
    print 'done!'
    print

  # 
  print 'Preparing training data...'
  ardl_roidb.prepare_roidb(imdb, cfg.TRAIN_PHASE)
  print 'done!'
  print

  return imdb.roidb


# 
def train_net(solver_prototxt, roidb, output_dir,
            pretrained_model=None, max_iters=40000, \
            exper_name=None, re_iter=None):
  """Train a Fast R-CNN network."""
  if exper_name:
    output_dir = os.path.join(output_dir, exper_name)
    print
    print "output_dir (final):", output_dir
    print
  # 
  sw = SolverWrapper(solver_prototxt, roidb, output_dir,
                     pretrained_model=pretrained_model, \
                     re_iter=re_iter)

  print
  print 'Solving...'
  print
  sw.train_model(max_iters)
  print
  print 'Done solving...'
  print
