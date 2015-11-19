# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# 
# Modified by ddk
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""

__sets = {}

import action_datasets.rcnn_action
import numpy as np
import sys


def _selective_search_IJCV_top_k(datatype, split, top_k):
    """Return an imdb that uses the top k proposals from the selective search
    IJCV code.
    """
    imdb = action_datasets.rcnn_action(datatype, split)
    imdb.roidb_handler = imdb.selective_search_IJCV_roidb
    imdb.config['top_k'] = top_k
    return imdb


# Set up rcnn_<>_<> using selective search "fast" mode
# For some action action_datasets, like PascalVoc2012, willowactions, stanford, mpii or more
for datatype in ["PascalVoc2012", "Willowactions", "Stanford", "MPII"]:
    for split in ["train", "val", "trainval", "test"]:
        name = "rcnn_{}_{}".format(datatype, split)
        __sets[name] = (lambda datatype= datatype, split=split:
                action_datasets.rcnn_action(datatype, split))


# Set up rcnn_<>_<>_top_<k> using selective search "quality" mode
# For some action action_datasets, like willowactions, stanford, mpii or more
# but only returning the first k boxes
for top_k in np.arange(1000, 11000, 1000):
    for datatype in ["PascalVoc2012", "Willowactions", "Stanford", "MPII"]:
        for split in ["train", "val", "trainval", "test"]:
            name = "voc_{}_{}_top_{:d}".format(datatype, split, top_k)
            __sets[name] = (lambda datatype= datatype, split=split, top_k=top_k:
                    _selective_search_IJCV_top_k(datatype, split, top_k))


# 
# ######################################################################
# 


# return an instance of dataset.rcnn_action which can get the imdb 
def get_imdb(name):
    """Get an imdb (image database) by name."""
    if not __sets.has_key(name):
        raise KeyError('Unknown dataset: {}'.format(name))

    # __sets[name](): <action_datasets.rcnn_action.rcnn_action object>
    print "name-origin:", name
    print
    return __sets[name]()


# return the keys, like {rcnn_<>_<>}
def list_imdbs():
    """List all registered imdbs."""
    return __sets.keys()
