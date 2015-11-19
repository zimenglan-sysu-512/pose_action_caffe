  // Copyright 2015 Zhu.Jin Liang

#ifndef CAFFE_CHAR_RECOGNITION_LAYERS_HPP_
#define CAFFE_CHAR_RECOGNITION_LAYERS_HPP_

#include <string>
#include <utility>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "hdf5.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/net.hpp"
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/data_layers.hpp"
#include "caffe/loss_layers.hpp"
#include "caffe/internal_thread.hpp"

using std::vector;
using std::string;

namespace caffe {

// data/images
// labels/coords
// aux info (img_ind, width, height, im_scale)
// support multi-source and multi-scale
template <typename Dtype>
class CharImageDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit CharImageDataLayer(const LayerParameter& param)
      : BasePrefetchingDataLayer<Dtype>(param) {}
  virtual ~CharImageDataLayer();
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "CharImageData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 3; }

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

 protected:
  shared_ptr<Caffe::RNG> prefetch_rng_;
  virtual void ShuffleImages();
  virtual void InternalThreadEntry();

  bool shuffle_;
  bool is_color_;
  bool is_scale_image_;
  bool always_max_size_;
  bool has_mean_values_;

  int lines_id_;
  int max_size_;
  int label_num_;
  int crop_size_;
  int batch_size_;
  std::vector<int> min_sizes_;
  std::string img_ext_;
  std::vector<std::string> sources_;
  std::vector<std::string> root_folders_;
  std::vector<std::string> objidxs_;
  std::vector<std::string> imgidxs_;
  std::vector<std::string> images_paths_;
  vector<Dtype> mean_values_;
  // <imgidx, <objidx, label>>
  vector<std::pair<int, std::pair<std::string, std::pair<std::string, int > > > >lines_;
  Blob<Dtype> aux_info_;
};

template <typename Dtype>
class CharAccuracyLayer : public Layer<Dtype> {
 public:
  explicit CharAccuracyLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "CharAccuracy"; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  /// @brief Not implemented -- CharAccuracyLayer cannot be used as a loss.
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    for (int i = 0; i < propagate_down.size(); ++i) {
      if (propagate_down[i]) { NOT_IMPLEMENTED; }
    }
  }
  int top_k_;
  int test_images_num_;
  int test_images_iter_;
  int test_images_counter_;
  /// Whether to ignore instances with a certain label.
  bool has_ignore_label_;
  /// The label indicating that an instance should be ignored.
  int ignore_label_;
  // 
  int label_axis_, outer_num_, inner_num_;
  int max_acc_iter_;
  Dtype max_acc_;
};

} // namespace caffe

#endif  // CAFFE_CHAR_RECOGNITION_LAYERS_HPP_