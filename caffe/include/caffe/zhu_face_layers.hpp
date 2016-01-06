// Copyright 2015 Zhu.Jin Liang

#ifndef CAFFE_ZHU_FACE_LAYERS_HPP_
#define CAFFE_ZHU_FACE_LAYERS_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/loss_layers.hpp"
#include "caffe/util/coords.hpp" 
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/* 
 * @brief bottom的size至少为3，
 * 				bottom[2...n]跟bottom[0]的长和宽要严格一致
 * 				先计算得到bottom[0]到bottom[1]的映射关系
 * 				对bottom[2...n]根据映射关系进行resize
 *
 * 				映射关系正常来讲是等于2的，一个表示x方向的映射，
 * 				一个是表示y方向的映射
 * 				如果映射关系的大小大于2，表示前面发生过截取多个patch拼接的情况，
 * 				这时候就需要按照不同的patch的映射关系进行映射，
 * 				一般来说是按channel划分。
 * 				比如映射关系为6的时候，表示有三种不同的x/y映射，
 * 				这时候bottom的channels数一定是3的倍数，假设是9，
 * 				那么channel 0~2用第1种x/y映射
 * 					 channel 3~5用第2种x/y映射
 * 					 channel 6~8用第3种x/y映射
 */
template <typename Dtype>
class ResizeWithMapLayer : public Layer<Dtype> {
public:
 explicit ResizeWithMapLayer(const LayerParameter& param)
     : Layer<Dtype>(param) {}
 virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
 virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
     const vector<Blob<Dtype>*>& top);

 virtual inline const char* type() const { return "ResizeWithMap"; }
 virtual inline int MinBottomBlobs() const { return 3; }
 virtual inline int MinTopBlobs() const { return 1; }
 virtual inline DiagonalAffineMap<Dtype> coord_map() {
   return DiagonalAffineMap<Dtype>(coefs_).inv();
 }

protected:
 virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
     const vector<Blob<Dtype>*>& top);
 virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
     const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
 virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
     const vector<Blob<Dtype>*>& top);
 virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
     const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

 // 表示bottom[1]到bottom[0]的变换
 vector<pair<Dtype, Dtype> > coefs_;

 enum Coefs_Mode {

	 // 表示coefs_只有1个
	 SINGLE_SAMPLE_SINGLE_CHANNEL = 0,

	 // 表示所有样本共享coefs，但不同channels的height*width的coefs不一样。
	 SINGLE_SAMPLE_MULTI_CHANNEL = 1,

	 // 表示每个样本都有独立的coefs，且不同channels的height*width的coefs不一样。
	 MULTI_SAMPLE_MULTI_CHANNEL = 2
 };
 Coefs_Mode coefs_mode_;

 // weights_跟locs_的四个维度分别表示：
 // num: 表示有多少个不同的样本的coefs
 // channels: 多少个x/y的映射关系;
 // height: 输出的map的一个channels的维度(bottom[1]->height() * bottom[1]->width())
 // width: 表示多少近邻（双线性插值是四近邻）
 // 							  这时候locs_表示输入的下标，weights_表示该点的权值
 Blob<Dtype> weights_;
 Blob<int> locs_;
 bool first_info_;
};

/*
 * @brief bottom的size至少为3，
 * 				bottom[2...n]跟bottom[0]的长和宽要严格一致
 * 				先计算得到bottom[1]到bottom[0]的映射关系
 * 				然后对于不同的channel，求出bottom[1]在height*width上的最大值的位置h, w
 * 				把h, w映射到bottom[1]的下h', w'，
 * 				然后再以h', w'为中心bottom[2...n]截patch
 * 				如果patch超出bottom[0]范围则补0
 */
template <typename Dtype>
class CropPatchFromMaxFeaturePositionLayer : public Layer<Dtype> {
 public:
  explicit CropPatchFromMaxFeaturePositionLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "CropPatchFromMaxFeaturePosition"; }
  virtual inline int MinBottomBlobs() const { return 3; }
  virtual inline int MinTopBlobs() const { return 1; }

  // 以当前crop的状态返回
  virtual inline DiagonalAffineMap<Dtype> coord_map() {
    CHECK_GT(crop_beg_.count(), 0) 
        << "Forward the net before calling this function";
    vector<pair<Dtype, Dtype> > coefs;
    for (int c = 0; c < crop_beg_.count(); c += 2) {
			coefs.push_back(make_pair(1, -crop_beg_.cpu_data()[c]));
			coefs.push_back(make_pair(1, -crop_beg_.cpu_data()[c + 1]));
    }
    return DiagonalAffineMap<Dtype>(coefs);
  }

  /* define in include/caffe/layer.hpp
  virtual inline DiagonalAffineMap<Dtype> coord_map() {
    // LOG(INFO) << "layer.hpp coord_map";
    NOT_IMPLEMENTED;
    // suppress warnings
    return DiagonalAffineMap<Dtype>(vector<pair<Dtype, Dtype> >());
  }
  */
 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Init(const vector<Blob<Dtype>*>& bottom, 
      const vector<Blob<Dtype>*>& top);
  virtual void DeriveCropBeg(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  // 表示bottom[1]到bottom[0]的变换
  Blob<Dtype> coefs_;
  int crop_w_, crop_h_;
  Blob<int> crop_beg_;
  // is_match_channel_ = 0: 截出来的patch会包括input1的所有channels
  // is_match_channel_ = 1: input2在channel n上的最大值只会用来截取input1 channel n的最大值
  Blob<int> is_match_channel_;
};

/* using top k 
 * @brief bottom的size至少为3，
 *        bottom[2...n]跟bottom[0]的长和宽要严格一致
 *        先计算得到bottom[1]到bottom[0]的映射关系
 *        然后对于不同的channel，求出bottom[1]在height*width上的最大值的位置h, w
 *        把h, w映射到bottom[1]的下h', w'，
 *        然后再以h', w'为中心bottom[2...n]截patch
 *        如果patch超出bottom[0]范围则补0
 */
template <typename Dtype>
class CropPatchFromMaxFeatureFromTopKPositionsLayer : public Layer<Dtype> {
 public:
  explicit CropPatchFromMaxFeatureFromTopKPositionsLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "CropPatchFromMaxFeatureFromTopKPositions"; }
  virtual inline int MinBottomBlobs() const { return 3; }
  virtual inline int MinTopBlobs() const { return 1; }

  // 以当前crop的状态返回
  virtual inline DiagonalAffineMap<Dtype> coord_map() {
    CHECK_GT(crop_beg_.count(), 0) 
        << "Forward the net before calling this function";
    vector<pair<Dtype, Dtype> > coefs;
    for (int c = 0; c < crop_beg_.count(); c += 2) {
      coefs.push_back(make_pair(1, -crop_beg_.cpu_data()[c]));
      coefs.push_back(make_pair(1, -crop_beg_.cpu_data()[c + 1]));
    }
    return DiagonalAffineMap<Dtype>(coefs);
  }

  /* define in include/caffe/layer.hpp
  virtual inline DiagonalAffineMap<Dtype> coord_map() {
    // LOG(INFO) << "layer.hpp coord_map";
    NOT_IMPLEMENTED;
    // suppress warnings
    return DiagonalAffineMap<Dtype>(vector<pair<Dtype, Dtype> >());
  }
  */
 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Init(const vector<Blob<Dtype>*>& bottom, 
      const vector<Blob<Dtype>*>& top);
  virtual void DeriveCropBeg(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void DeriveCropBegFromTopK(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  // so far we only use the highest score to get the corase predicted part location
  // and modify it to be get top k highest scores to crop the image, widh nms
  int top_k_;
  int nms_x_;
  int nms_y_;
  // 表示bottom[1]到bottom[0]的变换
  Blob<Dtype> coefs_;
  int crop_w_, crop_h_;
  Blob<int> crop_beg_;
  // is_match_channel_ = 0: 截出来的patch会包括input1的所有channels
  // is_match_channel_ = 1: input2在channel n上的最大值只会用来截取input1 channel n的最大值
  Blob<int> is_match_channel_;
};


/**
 * @brief Computes the Euclidean (L2) loss @f$
 *          E = \frac{1}{2N} \sum\limits_{n=1}^N \left| \left| \hat{y}_n - y_n
 *        \right| \right|_2^2 @f$ for real-valued regression tasks.
 *        跟普通的MSE比，差别在于他会根据选择性的传loss，一般来讲，设置了一个长度为n值，
 *        如果目标连续n个值都不合理，那么这部分的梯度不往回传。
 *        这个很有效，特别用于
 * @param bottom input Blob vector (length 2)
 *   -# @f$ (N \times C \times H \times W) @f$
 *      the predictions @f$ \hat{y} \in [-\infty, +\infty]@f$
 *   -# @f$ (N \times C \times H \times W) @f$
 *      the targets @f$ y \in [-\infty, +\infty]@f$
 * @param top output Blob vector (length 1)
 *   -# @f$ (1 \times 1 \times 1 \times 1) @f$
 *      the computed Euclidean loss: @f$ E =
 *          \frac{1}{2n} \sum\limits_{n=1}^N \left| \left| \hat{y}_n - y_n
 *        \right| \right|_2^2 @f$
 *
 * This can be used for least-squares regression tasks.  An InnerProductLayer
 * input to a EuclideanLossLayer exactly formulates a linear least squares
 * regression problem. With non-zero weight decay the problem becomes one of
 * ridge regression -- see src/caffe/test/test_sgd_solver.cpp for a concrete
 * example wherein we check that the gradients computed for a Net with exactly
 * this structure match hand-computed gradient formulas for ridge regression.
 *
 * (Note: Caffe, and SGD in general, is certainly \b not the best way to solve
 * linear least squares problems! We use it only as an instructive example.)
 *
 */
template <typename Dtype>
class EuclideanMaskLossLayer : public EuclideanLossLayer<Dtype> {
 public:
  explicit EuclideanMaskLossLayer(const LayerParameter& param)
      : EuclideanLossLayer<Dtype>(param) {}

  virtual inline const char* type() const { return "EuclideanMaskLoss"; }

 protected:
  /// @copydoc EuclideanLossLayer
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
};

/**
 * @brief Computes the Euclidean (L2) loss @f$
 *          E = \frac{1}{2N} \sum\limits_{n=1}^N \left| \left| \hat{y}_n - y_n
 *        \right| \right|_2^2 @f$ for real-valued regression tasks.
 *
 * @param bottom input Blob vector (length 2)
 *   -# @f$ (N \times C \times H \times W) @f$
 *      the predictions @f$ \hat{y} \in [-\infty, +\infty]@f$
 *   -# @f$ (N \times C \times H \times W) @f$
 *      the targets @f$ y \in [-\infty, +\infty]@f$
 * @param top output Blob vector (length 1)
 *   -# @f$ (1 \times 1 \times 1 \times 1) @f$
 *      the computed Euclidean loss: @f$ E =
 *          \frac{1}{2n} \sum\limits_{n=1}^N \left| \left| \hat{y}_n - y_n
 *        \right| \right|_2^2 @f$
 *
 * This can be used for least-squares regression tasks.  An InnerProductLayer
 * input to a EuclideanSelectiveLossLayer exactly formulates a linear least squares
 * regression problem. With non-zero weight decay the problem becomes one of
 * ridge regression -- see src/caffe/test/test_sgd_solver.cpp for a concrete
 * example wherein we check that the gradients computed for a Net with exactly
 * this structure match hand-computed gradient formulas for ridge regression.
 *
 * (Note: Caffe, and SGD in general, is certainly \b not the best way to solve
 * linear least squares problems! We use it only as an instructive example.)
 */
template <typename Dtype>
class EuclideanSelectiveLossLayer : public LossLayer<Dtype> {
 public:
  explicit EuclideanSelectiveLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param), diff_(), mask_() {}
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "EuclideanSelectiveLoss"; }

  virtual inline int ExactNumBottomBlobs() const { return -1; }
  /**
   * Unlike most loss layers, in the EuclideanSelectiveLossLayer we can backpropagate
   * to both inputs -- override to return true and always allow force_backward.
   */
  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return true;
  }

 protected:
  /// @copydoc EuclideanSelectiveLossLayer
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  /**
   * @brief Computes the Euclidean error gradient w.r.t. the inputs.
   *
   * Unlike other children of LossLayer, EuclideanSelectiveLossLayer \b can compute
   * gradients with respect to the label inputs bottom[1] (but still only will
   * if propagate_down[1] is set, due to being produced by learnable parameters
   * or if force_backward is set). In fact, this layer is "commutative" -- the
   * result is the same regardless of the order of the two bottoms.
   *
   * @param top output Blob vector (length 1), providing the error gradient with
   *      respect to the outputs
   *   -# @f$ (1 \times 1 \times 1 \times 1) @f$
   *      This Blob's diff will simply contain the loss_weight* @f$ \lambda @f$,
   *      as @f$ \lambda @f$ is the coefficient of this layer's output
   *      @f$\ell_i@f$ in the overall Net loss
   *      @f$ E = \lambda_i \ell_i + \mbox{other loss terms}@f$; hence
   *      @f$ \frac{\partial E}{\partial \ell_i} = \lambda_i @f$.
   *      (*Assuming that this top Blob is not used as a bottom (input) by any
   *      other layer of the Net.)
   * @param propagate_down see Layer::Backward.
   * @param bottom input Blob vector (length 2)
   *   -# @f$ (N \times C \times H \times W) @f$
   *      the predictions @f$\hat{y}@f$; Backward fills their diff with
   *      gradients @f$
   *        \frac{\partial E}{\partial \hat{y}} =
   *            \frac{1}{n} \sum\limits_{n=1}^N (\hat{y}_n - y_n)
   *      @f$ if propagate_down[0]
   *   -# @f$ (N \times C \times H \times W) @f$
   *      the targets @f$y@f$; Backward fills their diff with gradients
   *      @f$ \frac{\partial E}{\partial y} =
   *          \frac{1}{n} \sum\limits_{n=1}^N (y_n - \hat{y}_n)
   *      @f$ if propagate_down[1]
   */
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  Blob<Dtype> diff_;
  Blob<Dtype> mask_;
  std::vector<std::pair<Dtype, int> > neg_idx;

  // 强制的另label二值化
  Blob<Dtype> threshold_top_;
  bool force_binary_;
  Dtype binary_threshold_;
};

}  // namespace caffe

#endif  // CAFFE_ZHU_FACE_LAYERS_HPP_
