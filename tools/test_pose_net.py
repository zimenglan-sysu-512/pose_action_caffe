#!/usr/bin/env python
# -*- coding:utf-8 -*-

# --------------------------------------------------------
# Pose Estimation Demo
# Copyright (c) 2016 SYSU
# Written by Dengke Dong
# --------------------------------------------------------

"""Test a pose estimation network on images ."""

import _init_paths
import caffe
from fast_rcnn.config_pose import pose_cfg, cfg_from_file
from fast_rcnn.test_pose   import pose_estimation
import cv2
import time
import pprint
import socket
import os, sys
import argparse
import datetime


def _parse_args():
  """Parse input arguments"""
  parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
  # gpu
  parser.add_argument('--gpu_id', dest='gpu_id', help='gpu id',
                      default=0, type=int)
  # def
  parser.add_argument('--deploy_pt', dest='deploy_pt', help='pt file defining the network',
                      default="", type=str)
  # net
  parser.add_argument('--caffemodel', dest='caffemodel', help='trained model to test',
                      default="", type=str)
  # cfg
  parser.add_argument('--cfg_file', dest='cfg_file', help='optional config file',
                      default="", type=str)
  # pt_file
  parser.add_argument('--pt_file', dest='pt_file', help='person and torso detection results',
                      default="", type=str)
  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)
  args = parser.parse_args()
  return args

def _pt_bboxes(pt_file, l_pt_box=9):
  pt_infos = []
  fh       = open(pt_file)
  
  for line in fh.readlines():
    line = line.strip()
    info = line.split()
    assert len(info) >= 1
    im_path, info = info[0], info[1:]

    n_info = len(info)
    assert n_info >= l_pt_box
    assert n_info % l_pt_box == 0

    info     = [i.strip() for i in info]
    info     = [int(i)    for i in info]
    
    pt_boxes = []
    for j in xrange(n_info / l_pt_box):
      pt_box = []
      j2     = j * l_pt_box
      # ignore objidx
      j2     = j2 + 1
      pt_box.append(info[j2: j2+4])
      j2     = j2 + 4
      pt_box.append(info[j2: j2+4])
      pt_boxes.append(pt_box)

    pt_infos.append((im_path, pt_boxes))

  fh.close()
  assert len(pt_infos) >= 1

  return pt_infos

def _init_net():
  args = _parse_args()
  print "\n\n", args, "\n\n"

  cfg_file = args.cfg_file.strip()
  if not os.path.exists(cfg_file) or not os.path.isfile(cfg_file):
    raise IOError(('{:s} not found.\n').format(cfg_file))
  cfg_from_file(args.cfg_file)
  
  print('Using config:')
  pprint.pprint(pose_cfg)
  print "\n\n"
  
  time.sleep(pose_cfg.SLEEP_TIME)

  caffe.set_mode_gpu()
  caffe.set_device(args.gpu_id)

  deploy_pt  = args.deploy_pt.strip()
  if not os.path.exists(deploy_pt) or not os.path.isfile(deploy_pt):
    raise IOError(('{:s} not found.\n').format(deploy_pt))

  caffemodel = args.caffemodel.strip()
  if not os.path.exists(caffemodel) or not os.path.isfile(caffemodel):
    raise IOError(('{:s} not found.\n').format(caffemodel))
  net = caffe.Net(deploy_pt, caffemodel, caffe.TEST)

  return net, args

def test_pose_net():
  ''''''
  net, args = _init_net()

  pt_file = args.pt_file.strip()
  if not os.path.exists(pt_file) or not os.path.isfile(pt_file):
    raise IOError(('{:s} not found.\n').format(pt_file))
  # get info of person and torso detection results
  pt_infos = _pt_bboxes(pt_file)

  # #################################################################
  print "\n\nStarting pose estimation\n\n"

  im_c = 1
  for pt_info in pt_infos:
    if pose_cfg.DISP_NUM > 0 and im_c > pose_cfg.DISP_NUM:
      break
    
    im_path, pt_boxes = pt_info
    print "\n\n", "im_c:", im_c, "im_path:", im_path
    
    pose_estimation(net, im_path, pt_boxes)

    im_c = im_c + 1
    
  print "\n\nEnd pose estimation\n\n"

def _init_socket():
  # buffer size
  BUF_SIZE = pose_cfg.SOCKET.BUFFER_SIZE
  # ip and port  
  server_addr = (pose_cfg.SOCKET.SERVER_ADDR, pose_cfg.SOCKET.SERVER_PORT)

  # socket instance
  try :
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  except socket.error, msg :
    print "Creating Socket Failure. Error Code : " + str(msg[0]) + " Message : " + msg[1]
    sys.exit()
  print "Socket Created!"

  #设置地址复用
  server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

  try : 
    # 绑定地址
    server.bind(server_addr)
  except socket.error, msg :
    print "Binding Failure. Error Code : " + str(msg[0]) + " Message : " + msg[1]
    sys.exit()
  print "Socket Bind!"

  #监听, 最大监听数为5
  server.listen(5)
  print "Socket listening"

  return server

def _per_pt_info(line, l_pt_box=9):
  line = line.strip()
  info = line.split()
  assert len(info) >= 1
  im_path, info = info[0], info[1:]

  n_info = len(info)
  assert n_info >= l_pt_box
  assert n_info % l_pt_box == 0

  info     = [i.strip() for i in info]
  info     = [int(i)    for i in info]
  
  pt_boxes = []
  for j in xrange(n_info / l_pt_box):
    pt_box = []
    j2     = j * l_pt_box
    # ignore objidx
    j2     = j2 + 1
    pt_box.append(info[j2: j2+4])
    j2     = j2 + 4
    pt_box.append(info[j2: j2+4])
    pt_boxes.append(pt_box)

  assert len(pt_boxes) >= 1
  return im_path, pt_boxes

def _per_pt_info2(line, l_pt_box=13):
  # p_bbox: produce by t_bbox
  # p_bbox: produce by person detector
  # t_bbox: produce by toros  detector
  line = line.strip()
  info = line.split()
  assert len(info) >= 1
  im_path, info = info[0], info[1:]

  n_info = len(info)
  assert n_info >= l_pt_box
  assert n_info % l_pt_box == 0

  info     = [i.strip() for i in info]
  info     = [int(i)    for i in info]
  
  pt_boxes = []
  for j in xrange(n_info / l_pt_box):
    pt_box = []
    j2     = j * l_pt_box
    # ignore objidx
    j2     = j2 + 1
    pt_box.append(info[j2: j2+4])
    j2     = j2 + 4
    pt_box.append(info[j2: j2+4])
    j2     = j2 + 4
    pt_box.append(info[j2: j2+4])
    pt_boxes.append(pt_box)

  assert len(pt_boxes) >= 1
  return im_path, pt_boxes

def server_pose_net():
  net, args = _init_net()

  server = _init_socket()

  while True:
    #接收TCP连接, 并返回新的套接字和地址, 阻塞函数
    client, client_addr = server.accept()  
    print 'Connected by', client_addr
    BUF_SIZE = pose_cfg.SOCKET.BUFFER_SIZE
    while True:
      try: 
        # recieve data from client
        data = client.recv(BUF_SIZE)
        data = data.strip()
        print "\ndata recieve:", data, "\n"

        # pose estimation
        starttime = time.time()

        # im_path, pt_boxes = _per_pt_info(data)
        im_path, pt_boxes = _per_pt_info2(data)

        out_path = pose_estimation(net, im_path, pt_boxes)

        endtime = time.time()

        # send data to client
        data = out_path + " " + '%.3f' % (endtime - starttime)
        print "data send:", data
        print "\n*********************************************\n\n"
        client.sendall(data)
      except Exception as err:
        print "cant not get person & torso detection results"

  # close
  server.close()

if __name__ == '__main__':
  ''''''
  # # from pt file
  # test_pose_net()

  # # from client
  server_pose_net()
