// ------------------------------------------------------------------
// Fast R-CNN
// Copyright (c) 2015 Microsoft
// Licensed under The MIT License [see fast-rcnn/LICENSE for details]
// Written by DDK
// ------------------------------------------------------------------

#ifndef CAFFE_FAST_RCNN_ACTION_LAYERS_HPP_
#define CAFFE_FAST_RCNN_ACTION_LAYERS_HPP_

#include <vector>
#include <string>
#include <utility>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/neuron_layers.hpp"
#include "caffe/loss_layers.hpp"
#include "caffe/data_layers.hpp"

namespace caffe {

/**
 * @brief Takes one Blobs as input, output at least two Blob%s 
 * and split the bottom blob  along the num dimension
 * Used for ROIPooling to extract the primary region and the secondary region sets
 */
template <typename Dtype>
class CrossBlobsSumLayer : public Layer<Dtype> {
 public:
  explicit CrossBlobsSumLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "CrossBlobsSum"; }
  virtual inline int MinBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
};

/**
 * @brief Takes one Blobs as input, output at least two Blob%s 
 * and split the bottom blob  along the num dimension
 * Used for ROIPooling to extract the primary region and the secondary region sets
 */
template <typename Dtype>
class ExtractPrimaryLayer : public Layer<Dtype> {
 public:
  explicit ExtractPrimaryLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "ExtractPrimary"; }
  // virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int MinBottomBlobs() const { return 1; }
  virtual inline int MaxBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 2; }

 protected:
  
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

 	int ims_per_batch_;
};

/**
 * @brief Takes one Blobs as input, output one Blob 
 * Used for ROIPooling to extract the max-score of secondary region sets
 */
template <typename Dtype>
class SecondaryRegionsOpScoresLayer : public Layer<Dtype> {
 public:
  explicit SecondaryRegionsOpScoresLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "SecondaryRegionsOpScores"; }
  // bottom[0]: secondary regions' scores, like fc8, the pre-layer of softmax layer
  // bottom[1]: n_rois_regions (the number of secondary regions of per image, produced by data layer)
  virtual inline int MinBottomBlobs() const { return 1; }
  virtual inline int MaxBottomBlobs() const { return 2; }
  // 
  virtual inline int MinTopBlobs() const { return 1; }
  // MAX POOL layers can output an extra top blob for the mask;
  // others can only output the pooled inputs.
  // top[0]: secondary_max_score
  // top[1]: max_selected_secondary_regions_inds, if have
  //    record the index of selected secondary region, for regions extractions
  //    as input to `FusionRegionsLayer
  // top[2]: secondary_max_score_inds, if have
  //    record the pos of selected secondary regions in the blob, for back-propogation
  virtual inline int MaxTopBlobs() const {
    return (this->layer_param_.secondary_regions_op_scores_param().op_scores() ==
            SecondaryRegionsOpScoresParameter_OpScoresMethod_MAX) ? 3 : 1;
  }

 protected:
  
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  bool has_max_op_;
  bool init_first_;

  int ims_per_batch_;
  int n_secondary_regions_;
  Blob<int> max_idx_;
};

// idx (ims_per_batch, n_classes, 1 , 1)
// primary regions (ims_per_batch, 5, 1, 1)
// secondary regions (n_secondary_regions * ims_per_batch, 5, 1, 1)
// labels (ims_per_batch, 1, 1, 1)
template <typename Dtype>
class FusionRegionsLayer : public Layer<Dtype> {
 public:
  explicit FusionRegionsLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "FusionRegions"; }
  // bottom[0]: max_selected_secondary_regions_inds
  // bottom[1]: primary regions
  // bottom[2]: secondary regions
  // bottom[3]: labels
  virtual inline int ExactNumBottomBlobs() const { return 4; }
  // top[0]: fused_primary_regions (only ground truth class)
  // top[1]: fused_primary_regions (all classes), if have
  // top[2]: all corresponding max-selected secondary regions, if have
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 3; }

 protected:
  
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int n_classes_;
  int ims_per_batch_;
  Blob<Dtype> max_selected_secondary_regions_;
};


/* 
 * ScenePoolingLayer - spatial pyramid pooling on the whole image
*/
template <typename Dtype>
class ScenePoolingLayer : public Layer<Dtype> {
 public:
  explicit ScenePoolingLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "ScenePooling"; }

  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int num_;
  int channels_;
  int height_;
  int width_;
  int pooled_height_;
  int pooled_width_;
  Dtype spatial_scale_;
  Blob<int> max_idx_;
  Blob<Dtype> rois_blob_;
};

// Normalize the feature:
//  NF_L1(xi) = xi / (|x1| + ... + |xn|), where i = 1, ..., n
//  NF_L2(xi) = xi / (x1^2 + ... + xn^2), where i = 1, ..., n
//  NF_L1Sq(xi) = xi / sqrt(x1^2 + ... + xn^2), where i = 1, ..., n
//  We either regard each (1, channels, height, width) as a feature, 
//  or regard (num, channels, height, width) as a feature
// Normalizing the feature is independent.
template <typename Dtype>
class NormalizeFeatureLayer : public Layer<Dtype> {
 public:
  explicit NormalizeFeatureLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "NormalizeFeature"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  /// @copydoc NormalizeFeatureLayer
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int num_;
  int channels_;
  int height_;
  int width_;
  
  Blob<Dtype> fdot_sum_;
  Blob<Dtype> fsqrt_sum_;
  Blob<Dtype> multiplier_;

  NormalizeFeaturesParameter norm_feat_param_;
};

// Force the same label to be the more same
template <typename Dtype>
class SimilaryEuclideanLossLayer : public LossLayer<Dtype> {
 public:
  explicit SimilaryEuclideanLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param), diff_() {}
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "SimilaryEuclideanLoss"; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int MaxTopBlobs() const { return 1; }
  /**
   * Unlike most loss layers, in the SimilaryEuclideanLossLayer we can backpropagate
   * to both inputs -- override to return true and always allow force_backward.
   */
  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return true;
  }

 protected:
  /// @copydoc SimilaryEuclideanLossLayer
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  Blob<Dtype> diff_;
  Dtype learn_lr_;
  Dtype loss_thresh_;
  Dtype loss_bias_;
  bool above_loss_thresh_;
};

/**
 * @brief Pools the feature maps by taking the max, average, etc. within cubes.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class Pooling3DLayer : public Layer<Dtype> {
 public:
  explicit Pooling3DLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Pooling3D"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int MinTopBlobs() const { return 1; }
  // MAX POOL layers can output an extra top blob for the mask;
  // others can only output the pooled inputs.
  virtual inline int MaxTopBlobs() const {
    return (this->layer_param_.group_pooling_param().pool() ==
            PoolingParameter_PoolMethod_MAX) ? 2 : 1;
  }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  // virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
  //     const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  // virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
  //     const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int kernel_h_;
  int kernel_w_;
  int stride_h_;
  int stride_w_;
  int pad_h_;
  int pad_w_;
  int pooled_height_;
  int pooled_width_;
  bool global_pooling_;
  // 
  int num_;
  int channels_;
  int height_;
  int width_;
  // cube
  int pooled_channels_;
  int temporal_length_;
  int temporal_stride_;

  Blob<Dtype> rand_idx_;
  Blob<int> max_idx_;
};

}  // namespace caffe

#endif  // CAFFE_FAST_RCNN_ACTION_LAYERS_HPP_
