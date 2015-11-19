# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# 
# Modified by ddk
# --------------------------------------------------------

import action_datasets
import action_datasets.rcnn_action
import action_datasets.imdb
import utils.cython_bbox
from fast_rcnn.config_action import cfg

# 
import os
import sys
import cPickle
import subprocess
import numpy as np
import numpy.random as npr
import scipy.sparse
import scipy.io as sio
import xml.dom.minidom as minidom



# datatype: "PascalVoc2012", "Willowactions", "Stanford", "MPII"
# split/imageset: "train", "val", "trainval", "test" 
class rcnn_action(action_datasets.imdb):
  ''''''
  def __init__(self, datatype, imageset, data_dir=None):
    # 
    action_datasets.imdb.__init__(self, 'rcnn_' + datatype + '_' + imageset)
    # 
    self._datatype = datatype
    self._image_set = imageset
    self._data_dir = self._get_default_path() if data_dir is None \
                        else data_dir
    # data_dir: path to the action directory
    # data_path: path to the directory of specific action dataset
    # data/Actions/
    # data/Actions/{datatype}
    self._data_path = os.path.join(self._data_dir, datatype)
    self._data_path = self._data_path + os.sep
    print "data_path:", self._data_path
    print

    # read from file
    self._classes_filename = "action_classes.txt"
    self._classes_filepath = self._data_path + self._classes_filename
    print 
    print "classes_filepath:", self._classes_filepath
    print
    # Check
    if os.path.exists(self._classes_filepath):
      # Open and read
      with open(self._classes_filepath) as f:
        self._classes = [x.strip().lower() for x in f.readlines()]
      self._classes = tuple(self._classes)
    else:
      print "Missing classes_filepath"
      print "Here we use PascalVoc2012 actions classes ..."
      print
      # Default actions classes' name -- PascalVoc2012
      self._classes = ('other', # always index 0
                       'jumping', 'phoning', 'playinginstrument',
                       'reading', 'ridingbike', 'ridinghorse',
                       'running', 'takingphoto', 'usingcomputer',
                       'walking',)
    # for example
    # class_to_ind: {
    #   'other': 0, 
    #   'jumping': 1, 
    #   'phoning': 2, 
    #   'playinginstrument': 3, 
    #   'reading': 4,
    #   'ridingbike': 5,
    #   'ridinghorse': 6,
    #   'running': 7,
    #   'takingphoto': 8,
    #   'usingcomputer': 9,
    #   'walking': 10, }
    self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
    self._ind_to_class = dict(zip(xrange(self.num_classes), self.classes))
    # print self._class_to_ind

    # path to the annotation file: data/Actions/${datatype}/<imageset>_anno_file.txt
    self._anno_filename = "anno_file.txt"
    self._anno_filepath = self._data_path + imageset + "_" + self._anno_filename
    print 
    print "anno_filepath:", self._anno_filepath
    print

    # 
    self._image_ext = '.jpg'
    self._image_index = self._load_image_set_index(self._anno_filepath)

    # Default to roidb handler
    self._roidb_handler = self.selective_search_roidb

    # PASCAL specific config options
    self.config = {'cleanup'  : True,
                   'use_salt' : True,
                   'top_k'    : 2000,
                   'top_N'    : 10,  }
    # check
    assert os.path.exists(self._data_dir), \
            'Data directory does not exist: {}'.format(self._data_dir)
    assert os.path.exists(self._data_path), \
            'Data path does not exist: {}'.format(self._data_path)

  @property
  def class_to_ind(self):
    return self._class_to_ind
  @property
  def ind_to_class(self):
    return self._ind_to_class
  

    

  # 
  # #####################################################################
  # 

        
  def image_path_from_index(self, index):
    """
    Construct an image path from the image's "index" identifier.
    """
    image_path = os.path.join(self._data_path, 'JPEGImages',
                              index + self._image_ext)
    assert os.path.exists(image_path), \
            'Path does not exist: {}'.format(image_path)
    return image_path


  # Get the imagepath corresponding `imagename without ext`
  def image_path_at(self, i):
    """
    Return the absolute path to image i in the image sequence.
    """
    return self.image_path_from_index(self._image_index[i])


  # Action data directory: /home/black/caffe/fast-rcnn-action/data/Actions/
  # For specific action dataset, you can use `ln -s`
  # For example:
  #   cd .
  #   cd ${self._data_dir}
  #   ln -s "../pathTo/PascalVoc2012/" "PascalVoc2012"
  def _get_default_path(self):
      """
      Return the default path where PASCAL VOC is expected to be installed.
      """
      return os.path.join(action_datasets.ROOT_DIR, 'data', 'Actions')


  # 
  # #####################################################################
  # 

  
  # anno_filepath: data/Actions/${datatype}/anno_file.txt
  def _load_image_set_index(self, anno_filepath):
    """
    Load the indexes listed in this dataset's image set file.
    """
    # Check
    assert os.path.exists(anno_filepath), \
            'Path does not exist: {}'.format(anno_filepath)
    # Open and read
    with open(anno_filepath) as f:
      # format: imgidx x1 y1 x2 y2 label_list
      #   whre label list look like this: 0 0 0 0 1 0 0 (assume here has six action classes)
      image_index = [x.strip().split()[0] for x in f.readlines()]
    # 
    return image_index


  # anno_filepath: data/Actions/${datatype}/anno_file.txt
  def _load_action_annotation(self, anno_filepath):
    """
    Load image and bounding boxes info from 
    """
    # Check
    assert os.path.exists(anno_filepath), \
            'Path does not exist: {}'.format(anno_filepath)
    # Open and read
    lines = None
    with open(anno_filepath) as f:
      # format: imgidx x1 y1 x2 y2 label_list
      #   whre label list look like this: 0 0 0 0 1 0 0 (assume here has six action classes)
      lines = f.readlines()
     
    if not lines:
      print
      print "missing anno_filepath:", anno_filepath
      sys.exit(1)

    # init
    image_index, gt_roidb = [], []

    # Process
    for line in lines:
      # Initialize
      boxes = np.zeros((1, 4), dtype=np.uint16)
      gt_classes = np.zeros(1, dtype=np.int32)
      overlaps = np.zeros((1, self.num_classes), dtype=np.float32)

      line = line.strip().split()
      args = 0
      imgidx = line[args]
      image_index.append(imgidx)

      args += 1
      x1, y1, x2, y2 = line[args: args + 4]
      x1 = float(x1) - 1
      y1 = float(y1) - 1
      x2 = float(x2) - 1
      y2 = float(y2) - 1

      args += 4
      classname = line[args]
      cls = self._class_to_ind[classname.lower().strip()]

      gt_classes[0] = cls
      boxes[0, :] = [x1, y1, x2, y2]
      overlaps[0, cls] = 1.0
      overlaps = scipy.sparse.csr_matrix(overlaps)

      # 
      img_anno_dict = {
          'boxes' : boxes, 
          'gt_classes': gt_classes, 
          'gt_overlaps' : overlaps, 
          'flipped' : False}
      gt_roidb.append(img_anno_dict)

    return image_index, gt_roidb


  def gt_roidb(self):
    """
    Return the database of ground-truth regions of interest.

    This function loads/saves from/to a cache file to speed up future calls.
    """
    # cache_path: data/cache/Actions/
    # self.name: rcnn_<>_<>
    print "cache_path:", self.cache_path

    cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as fid:
            roidb = cPickle.load(fid)
        print '{} gt roidb loaded from {}'.format(self.name, cache_file)
        return roidb

    # Get image_index and gt_roidb
    # self._anno_filepath: data/Actions/${datatype}/anno_file.txt
    self._image_index, gt_roidb = self._load_action_annotation(self._anno_filepath)

    print "cache_file:", cache_file
    with open(cache_file, 'wb') as fid:
        cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
    print
    print 'wrote gt roidb to {}'.format(cache_file)

    return gt_roidb


    # ##################################################
    # 
    # ##################################################


  def _load_selective_search_roidb(self, gt_roidb):
    # cache_path: data/cache/Actions/${datatype}
    # self.name: rcnn_<>_<>
    # filename: data/Actions/${datatype}/rcnn_<datatype>_<imageset>.mat
    filename = os.path.abspath(os.path.join(self.cache_path, '..', '..', '..',
                                            'Actions', self._datatype, 
                                            self.name + '.mat'))
    print 'filename:', filename
    assert os.path.exists(filename), \
           'Selective search data not found at: {}'.format(filename)
    raw_data = sio.loadmat(filename)['boxes'].ravel()
    raw_imgidx = sio.loadmat(filename)['images'].ravel()

    box_list = []
    temp_box_list = []
    imgidx_dict = {}

    img_count = 0
    for i in xrange(raw_data.shape[0]):
      imgidx = raw_imgidx[i][0]
      imgidx = str(imgidx)
      if imgidx[-1] == ".":
        imgidx = imgidx[:-1]
      if imgidx not in imgidx_dict.keys():
        imgidx_dict[imgidx] = img_count
        temp_box_list.append(raw_data[i][:, (1, 0, 3, 2)] - 1)
        img_count = img_count + 1

    image_index = []
    gt_roidb2 = []
    imgidx_keys = imgidx_dict.keys();
    ignore_count = 0
    missing_count = 0

    image_index_len = 0
    if cfg.LESSEN_DEBUG_TIME:
      image_index_len = cfg.LESSEN_DEBUG_IMAGE_INDEX_LEN
    else:
      image_index_len = len(self._image_index)
    # 
    for idx in xrange(image_index_len):
      imgidx = self._image_index[idx]
      if imgidx in imgidx_keys:
        boxidx = imgidx_dict[imgidx]
        # Check
        if len(temp_box_list[boxidx]) < cfg.TRAIN.MIN_BATCH_SIZE_PER_IMAGE:
          # print "old shape:", temp_box_list[boxidx].shape
          ignore_count += 1
          real_bbox_len = temp_box_list[boxidx].shape[0]
          need_bbox_len = cfg.TRAIN.MIN_BATCH_SIZE_PER_IMAGE - real_bbox_len

          need_bbox_idxs = npr.choice(xrange(real_bbox_len), size=need_bbox_len,
                         replace=True)
          need_bboxes = [temp_box_list[boxidx][nbi] for nbi in need_bbox_idxs]
          temp_box_list[boxidx] = np.vstack((temp_box_list[boxidx], need_bboxes))
          # print "new shape:", temp_box_list[boxidx].shape

        # ####################################
        # reset
        image_index.append(imgidx)
        gt_roidb2.append(gt_roidb[idx])
        # ####################################

        box_list.append(temp_box_list[boxidx])
      else:
        print "*** error, missing image or labels:", imgidx
        missing_count += 1
    print 
    print "ignore_count:", ignore_count
    print "missing_count:", missing_count
    print 

    # reset because some missing
    self._image_index = image_index
    print "image_index.shape:", len(image_index)
    print "image_index.shape:", len(self._image_index)
    print "box_list.shape:", len(box_list)
    print "gt_roidb.shape:", len(gt_roidb)

    # use gt_roidb2 instead gt_roidb, maybe missing some
    ss_roidb = self.create_roidb_from_box_list(box_list, gt_roidb2)
    return gt_roidb2, ss_roidb


  def selective_search_roidb(self):
    """
    Return the database of selective search regions of interest.
    Ground-truth ROIs are also included.

    This function loads/saves from/to a cache file to speed up future calls.
    """
    # cache_path: data/cache/Actions/
    # self.name: rcnn_<>_<>
    # cache_file: data/cache/Actions/rcnn_<datatype>_<imageset>_selective_search_roidb.pkl
    cache_file = None
    if cfg.LESSEN_DEBUG_TIME:
      lessen_debug_str = cfg.LESSEN_DEBUG_STR
      cache_file = os.path.join(self.cache_path, self.name + "_" + 
          lessen_debug_str + '_selective_search_roidb.pkl')
    else:
      cache_file = os.path.join(self.cache_path, self.name + '_selective_search_roidb.pkl')

    if os.path.exists(cache_file):
      with open(cache_file, 'rb') as fid:
          roidb = cPickle.load(fid)
      print '{} ss roidb loaded from {}'.format(self.name, cache_file)
      return roidb

    if self._image_set != 'test':
      gt_roidb = self.gt_roidb()
      gt_roidb, ss_roidb = self._load_selective_search_roidb(gt_roidb)
      roidb = action_datasets.imdb.merge_roidbs(gt_roidb, ss_roidb)
    else:
      roidb = self._load_selective_search_roidb(None)


    with open(cache_file, 'wb') as fid:
      cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
    print 'wrote ss roidb to {}'.format(cache_file)

    return roidb


  # ##################################################
  # Get selective search IJCV roidb
  # ##################################################


  def _load_selective_search_IJCV_roidb(self, gt_roidb):
    # cache_path: data/cache/Actions/
    # self.name: rcnn_<>_<>
    # IJCV_path: data/Actions/${datatype}/rcnn_<datatype>_<imageset>_IJCV
    IJCV_path = os.path.abspath(os.path.join(self.cache_path, '..', '..',
                                             'Actions', self._datatype, 
                                             self.name + '_IJCV'))
    assert os.path.exists(IJCV_path), \
           'Selective search IJCV data not found at: {}'.format(IJCV_path)

    top_k = self.config['top_k']
    box_list = []
    # Get image_index
    if self.image_index is None:
      self.image_index = self._load_image_set_index(self._anno_filepath)

    for i in xrange(self.num_images):
      filename = os.path.join(IJCV_path, self.image_index[i] + '.mat')
      raw_data = sio.loadmat(filename)
      box_list.append((raw_data['boxes'][:top_k, :]-1).astype(np.uint16))

    return self.create_roidb_from_box_list(box_list, gt_roidb)


  def selective_search_IJCV_roidb(self):
    """
    Return the database of selective search regions of interest.
    Ground-truth ROIs are also included.

    This function loads/saves from/to a cache file to speed up future calls.
    """
    # cache_path: data/cache/Actions/
    # self.name: rcnn_<>_<>
    # cache_file: data/Actions//rcnn_<datatype>_<imageset>_selective_search_IJCV_top_<top_k>.pkl
    cache_file = os.path.join(self.cache_path, self._datatype,
            '{:s}_selective_search_IJCV_top_{:d}_roidb.pkl'.
            format(self.name, self.config['top_k']))

    if os.path.exists(cache_file):
      with open(cache_file, 'rb') as fid:
        roidb = cPickle.load(fid)
      print '{} ss roidb loaded from {}'.format(self.name, cache_file)
      return roidb

    gt_roidb = self.gt_roidb()
    ss_roidb = self._load_selective_search_IJCV_roidb(gt_roidb)
    # Merge
    roidb = action_datasets.imdb.merge_roidbs(gt_roidb, ss_roidb)

    with open(cache_file, 'wb') as fid:
      cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
    print 'wrote ss roidb to {}'.format(cache_file)

    return roidb


  def competition_mode(self, on):
    if on:
      self.config['use_salt'] = False
      self.config['cleanup'] = False
    else:
      self.config['use_salt'] = True
      self.config['cleanup'] = True



# 
if __name__ == '__main__':
  '''
  datatype: "PascalVoc2012", "Willowactions", "Stanford", "MPII"
  split: "train", "val", "trainval", "test" 
  '''
  d = action_datasets.rcnn_action('PascalVoc2012', 'trainval')
  res = d.roidb
  from IPython import embed
  embed()