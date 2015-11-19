// Copyright 2015 Zhu.Jin Liang

#ifndef CAFFE_WANGLAN_FACE_SHOULDERS_LAYERS_HPP_
#define CAFFE_WANGLAN_FACE_SHOULDERS_LAYERS_HPP_

#include <string>
#include <utility>
#include <vector>
#include <deque>
#include <iostream>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/data_layers.hpp"
#include "caffe/loss_layers.hpp"
#include "caffe/util/coords.hpp" 
#include "caffe/proto/caffe.pb.h"

using std::string;
using std::vector;
using std::pair;
using std::deque;

namespace caffe {

/*

This function is used to create a pthread that prefetches the window data.
  multiscale

window_multi_scale_data_param {
  # the file record all the samples, one line is one sample
  # format: filename(no postfix) x1 y1 x2 y2 ... xn yn
  # if there are several target in an image, just concat all the annotation in one line
  source: "./annotations/anno_3pts.anno"
  hard_sample_source: "./bootstrap/iter_60000.txt"
  batch_size: 1024
  
  # Fraction of batch that should be positive samples
  pos_fraction: 0.5

  patch_height: 49
  patch_width: 49

  # the folder stores all the images
  imgs_folder: "./dataSet/imageTrain"

  # the file records which key points to use
  # format: key_point_idx1 key_point_idx2 ... key_point_idxn
  # seperate by space, key_point_idx begins from 0 not 1 
  key_points_file: "./usedKeyPoints_all.txt"
  # specify the number of key points
  key_point_counts: 3

  # if this variable is set, the images are scaled according their bbox not some pairs of point
  standard_bbox_diagonal_len: 180

  # if is_range_scale is true, scales generate from [scale_lower_limit, scale_upper_limit]
  # else generate from the discrete scales recorded in scales_file
  is_range_scale: true
  scale_lower_limit: 2.3
  scale_upper_limit: 2.7
  # specify the scales to use
  # each scale seperate by a space
  #scales_file: "prototxt/face_geometry/conv_patch_no_mean_bootstrap_06_06/scales.txt"

  # if the distance between a point "pt1" and a key point "pt2" is not greater than valid_distance
  # set point "pt1" as a positive example of key point "pt2"
  valid_distance: 6
  invalid_distance: 24

  # Fraction of negative samples should be hard negative samples
  hard_sample_fraction: 0.7

  # assume the crop bbox is (x1, y1), (x2, y2)
  # if x1, y1, x2, y2 is out of image
  # set is_fill true: fill fill_value around image in order to keep ratio of crop bbox
  # set if_fill false: let x1 = MAX(0, x1), x2 = MIN(width, x2), the same as y, then crop the image
  is_fill: true
  fill_value: 0

  batch_reuse_count: 3

  visual_bbox_path: ""
}
*/
template <typename Dtype>
class WindowMultiScaleDataLayer: public BasePrefetchingDataLayer<Dtype> 
{
 public:
  explicit WindowMultiScaleDataLayer(const LayerParameter& param)
      : BasePrefetchingDataLayer<Dtype>(param) {}
  virtual ~WindowMultiScaleDataLayer();
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  //
  virtual inline const char* type() const { 
      return "WindowMultiScaleData"; 
  }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 3; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    return;
  }
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    return;
  }

 protected:
  virtual void ShuffleImages();
  virtual void InternalThreadEntry();
  virtual void PerturbedCoordsBias();
  virtual unsigned int PrefetchRand();
  // random seed
  shared_ptr<Caffe::RNG> prefetch_rng_;
  ///
  const deque<std::pair<std::string, vector<float> > >& hard_samples() {
    return hard_samples_;
  }
  deque<std::pair<std::string, vector<float> > >& mutable_hard_samples() {
    return hard_samples_;
  }
  const vector<std::pair<std::string, vector<float> > >& samples() {
    return samples_;
  }
  const vector<vector<float> >& bboxes_standard_scale() {
    return bboxes_standard_scale_;
  }
  
  bool with_bbox_;
  // scale generate from a range or some specified values
  bool is_range_scale_;
  float scale_lower_limit_;
  float scale_upper_limit_;
  // which scales should be used
  vector<float> scales_candidate_;
  shared_ptr<Blob<Dtype> > prefetch_data_;
  shared_ptr<Blob<Dtype> > prefetch_label_;
  shared_ptr<Blob<Dtype> > prefetch_bbox_;
  //
  vector<std::pair<std::string, vector<float> > > samples_;
  vector<std::pair<std::string, vector<int> > > samples_tags_;
  vector<vector<float> > bboxes_standard_scale_;
  vector<int> key_point_idxs_;
  // standard length:
  // The length between standard_len_[i][0] and standard_len_[i][1]
  // is expected to be standard_len_[i][2]
  vector<vector<int> > standard_len_;
  // first: filename
  // second: x1, y1, x2, y2
  deque<std::pair<std::string, vector<float> > > hard_samples_;

  int batch_reuse_count;
  int cur_batch_use_count;
};

template <typename Dtype>
class EuclideanLoss2Layer : public LossLayer<Dtype> {
 public:
  explicit EuclideanLoss2Layer(const LayerParameter& param)
      : LossLayer<Dtype>(param), diff_() {}
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "EuclideanLoss2"; }
  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return true;
  }

 protected:
  /// @copydoc EuclideanLoss2Layer
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  Blob<Dtype> diff_;
};

}  // namespace caffe

#endif  // CAFFE_WANGLAN_FACE_SHOULDERS_LAYERS_HPP_
