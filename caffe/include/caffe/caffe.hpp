// caffe.hpp is the header file that you need to include in your code. It wraps
// all the internal caffe header files into one for simpler inclusion.

#ifndef CAFFE_CAFFE_HPP_
#define CAFFE_CAFFE_HPP_

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/solver.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/vision_layers.hpp"

//
// global variables for each layers
#include "caffe/global_variables.hpp"

// 
// fast rcnn
#include "caffe/fast_rcnn_layers.hpp"
#include "caffe/fast_rcnn_action_layers.hpp"

// 
// zhu face
#include "caffe/zhu_face_layers.hpp"

// 
// pose estimation
#include "caffe/pose_estimation_layers.hpp"

//
// torso detection
#include "caffe/solver2.hpp"
#include "caffe/person_torso_layers.hpp"
#include "caffe/wanglan_face_shoulders_layers.hpp"

#endif  // CAFFE_CAFFE_HPP_
