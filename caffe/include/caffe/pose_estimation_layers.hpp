  // Copyright 2015 Zhu.Jin Liang

#ifndef CAFFE_POSE_ESTIMATION_LAYERS_HPP_
#define CAFFE_POSE_ESTIMATION_LAYERS_HPP_

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

// along channle do max-pooling op
template <typename Dtype>
class ArgMaxCLayer : public Layer<Dtype> {
 public:
  explicit ArgMaxCLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "ArgMaxC"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  /// @brief Not implemented
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  bool out_max_val_;
};

/**
 * @brief Computes the accuracy for human pose/joint estimation
 * Using Percentage of Detected Joints (PDJ)
 * We restrict that the coordinates are in this interval [0, width - 1] and [0, height - 1].
 * So if use regression and in the beginning normalize coordinates, you must re-normalize them,
 * before use this layer, be calling `revert_normalized_pose_coords_layer` layer.
 *
 * Please refer to  `Deeppose: Human Pose Estimation via Deep Neural Networks, CVPR 2014.`
 */
template <typename Dtype>
class PosePDJAccuracyLayer : public Layer<Dtype> {
 public:
  
  explicit PosePDJAccuracyLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "PosePDJAccuracy"; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
 virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  /// @brief Not implemented -- 
  /// PosePDJAccuracyLayer cannot be used as a loss.
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    for (int i = 0; i < propagate_down.size(); ++i) {
      if (propagate_down[i]) { NOT_IMPLEMENTED; }
    }
  }

  void initAccFactors();
  void InitQuantization();
  void CalAccPerImage(const Dtype* pred_coords_ptr, const Dtype* gt_coords_ptr);
  void WriteResults(const float total_accuracies[]);
  void QuanFinalResults();
  void Quantization(const Dtype* pred_coords, const Dtype* gt_coords, const int num);

 protected:
  bool zero_iter_test_;
  int label_num_;
  int images_num_;
  int key_point_num_;
  int images_itemid_;
  int acc_factor_num_;
  int shoulder_id_;
  int hip_id_;
  
  float acc_factor_;

  std::string acc_path_;
  std::string acc_name_;
  std::string acc_file_;
  std::string log_name_;
  std::string log_file_;

  std::vector<float> acc_factors_;
  std::vector< std::vector<float> > accuracies_;

  Blob<Dtype> diff_;
  std::vector<Dtype> max_score_;
  std::vector<Dtype> max_score_iter_;
};

/**
 * @brief Computes the accuracy for human pose/joint estimation
 * Using Percentage of Detected Joints (using euclidean distance)
 * We restrict that the coordinates are in this interval [0, width - 1] 
 * and [0, height - 1].
 * So if use regression and in the beginning normalize coordinates, 
 * you must re-normalize them,
 * before use this layer, be calling 
 * `revert_normalized_pose_coords_layer` layer.
 *
 * Please refer to  `Flowing ConvNets for Human Pose Estimation in Video, ICCV 2015.`
 */
template <typename Dtype>
class PoseEuDistAccuracyLayer : public Layer<Dtype> {
 public:
  
  explicit PoseEuDistAccuracyLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "PoseEuDistAccuracy"; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
 virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  /// @brief Not implemented -- 
  /// PoseEuDistAccuracyLayer cannot be used as a loss.
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, 
      const vector<Blob<Dtype>*>& bottom) {
    for (int i = 0; i < propagate_down.size(); ++i) {
      if (propagate_down[i]) { NOT_IMPLEMENTED; }
    }
  }

  void initAccFactors();
  void InitQuantization();
  void CalAccPerImage(const Dtype* pred_coords_ptr, 
                      const Dtype* gt_coords_ptr);
  void WriteResults(const float total_accuracies[]);
  void QuanFinalResults();
  void Quantization(const Dtype* pred_coords, 
                    const Dtype* gt_coords, 
                    const int num);

 protected:
  bool zero_iter_test_;
  int label_num_;
  int images_num_;
  int key_point_num_;
  int images_itemid_;
  int acc_factor_;
  int acc_factor_num_;
  
  std::string acc_path_;
  std::string acc_name_;
  std::string acc_file_;
  std::string log_name_;
  std::string log_file_;

  std::vector<float> acc_factors_;
  std::vector< std::vector<float> > accuracies_;

  Blob<Dtype> diff_;
  std::vector<Dtype> max_score_;
  std::vector<Dtype> max_score_iter_;
};

/**
 * @brief Computes the Euclidean (L2) loss @f$
 * bottom[0]: predicted heat maps
 * bottom[1]: ground truth heat maps
 * bottom[2]: ground truth masks (indicator for which heat maps are invalid)
 */
template <typename Dtype>
class PoseHeatMapLossLayer : public LossLayer<Dtype> {
 public:
  explicit PoseHeatMapLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param), diff_() {}
  virtual void LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

  virtual void Reshape(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual inline const char* type() const { return "PoseHeatMapLoss"; }
  virtual inline int MinBottomBlobs() const { return 2; }
  virtual inline int MaxBottomBlobs() const { return 3; }
  virtual inline int ExactNumTopBlobs() const { return 1; }
  /**
   * Unlike most loss layers, in the PoseHeatMapLossLayer we can backpropagate
   * to both inputs -- override to return true and always allow force_backward.
   */
  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return true;
  }

 protected:
  /// @copydoc PoseHeatMapLossLayer
  virtual void Forward_cpu(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  
  virtual Dtype PrintLoss();
  virtual void ComputesHeatMapLoss(const vector<Blob<Dtype>*>& bottom);
  virtual void CopyDiff(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  /// @copydoc PoseHeatMapLossLayer
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, 
      const vector<Blob<Dtype>*>& bottom);

  Dtype PrintLoss_gpu();
  /**
   * @brief Initialize the Random number generations if needed by the
   *    transformation.
   */
  void InitRand();
   /**
   * @brief Generates a random integer from Uniform({0, 1, ..., n-1}).
   *
   * @param n
   *    The upperbound (exclusive) value of the random number.
   * @return
   *    A uniformly random integer value from ({0, 1, ..., n-1}).
   */
  virtual int Rand(int n);
  void CheckRandNum_gpu();
  void ComputesHeatMapLoss_gpu(const vector<Blob<Dtype>*>& bottom);
  void CopyDiff_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  static bool cmp(const std::pair<int, Dtype>& a, 
      const std::pair<int, Dtype>& b) {
    return a.second > b.second;
  }
  bool has_ratio_;
  bool has_bg_num_;
  int bg_num_;
  int heat_num_;
  int key_point_num_;
  int loss_emphase_type_;
  Dtype fg_eof_, bg_eof_, ratio_;
  
  /// when divided by UINT_MAX, 
  /// the randomly generated values @f$u\sim U(0,1)@f$
  unsigned int uint_thres_;
  
  Blob<Dtype> diff_;
  Blob<unsigned int> rand_vec_;

  int prob_num_;
  int parts_err_num_thres_;
  float heat_score_thres_;
  vector<float> heat_score_thres_arr_;
  std::string hard_negative_filepath_;

  // shared_ptr<Caffe::RNG> prefetch_rng_;
  shared_ptr<Caffe::RNG> rng_;
};

// data/images
// labels/coords
// aux info (img_ind, width, height, im_scale)
template <typename Dtype>
class RandomImageDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit RandomImageDataLayer(const LayerParameter& param)
      : BasePrefetchingDataLayer<Dtype>(param) {}
  virtual ~RandomImageDataLayer();
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "RandomImageData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 3; }

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

 protected:
  shared_ptr<Caffe::RNG> prefetch_rng_;
  void PerturbedCoordsBias();
  virtual void ShuffleImages();
  virtual void InternalThreadEntry();

  bool shuffle_;
  bool is_color_;
  bool is_scale_image_;
  bool always_max_size_;
  bool has_mean_values_;
  bool is_perturb_coords_;

  int lines_id_;
  int min_size_;
  int max_size_;
  int label_num_;
  int crop_size_;
  int batch_size_;
  int heat_map_a_;
  int heat_map_b_;
  int key_points_num_;
  int receptive_field_size_;
  std::vector<int> origin_parts_orders_;
  std::vector<int> flippable_parts_orders_;

  float min_plb_;
  float max_plb_;

  std::string source_;
  std::string img_ext_;
  std::string root_folder_;
  std::string parts_orders_path_;

  // variables
  // std::vector<int> widths_;
  // std::vector<int> heights_;
  // std::vector<float> im_scales_;
  // std::vector<std::vector<float> > labels_;
  std::vector<std::string> objidxs_;
  std::vector<std::string> imgidxs_;
  std::vector<std::string> images_paths_;

  vector<Dtype> mean_values_;
  // <imgidx, <objidx, coords>>
  vector<std::pair<std::string, std::pair<std::string, std::vector<float> > > >lines_;
  
  Blob<Dtype> aux_info_;
  shared_ptr<Blob<Dtype> > perturbed_labels_bias_;  // default for coordinates
};

// data/images
// labels/coords
// aux info (img_ind, width, height, im_scale)
// support multi-source and multi-scale
template <typename Dtype>
class RandomImageData2Layer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit RandomImageData2Layer(const LayerParameter& param)
      : BasePrefetchingDataLayer<Dtype>(param) {}
  virtual ~RandomImageData2Layer();
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "RandomImageData2"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 3; }

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

 protected:
  shared_ptr<Caffe::RNG> prefetch_rng_;
  void PerturbedCoordsBias();
  virtual void ShuffleImages();
  virtual void InternalThreadEntry();

  bool shuffle_;
  bool is_color_;
  bool is_scale_image_;
  bool always_max_size_;
  bool has_mean_values_;
  bool is_perturb_coords_;

  int lines_id_;
  int max_size_;
  int label_num_;
  int crop_size_;
  int batch_size_;
  int key_points_num_;
  int receptive_field_size_;
  std::vector<int> min_sizes_;
  std::vector<int> origin_parts_orders_;
  std::vector<int> flippable_parts_orders_;

  float min_plb_;
  float max_plb_;

  std::string img_ext_;
  std::string parts_orders_path_;
  std::vector<std::string> sources_;
  std::vector<std::string> root_folders_;

  // variables
  std::vector<std::string> objidxs_;
  std::vector<std::string> imgidxs_;
  std::vector<std::string> images_paths_;

  vector<Dtype> mean_values_;
  // <imgidx, <objidx, coords>>
  vector<std::pair<int, std::pair<std::string, std::pair<std::string, std::vector<float> > > > >lines_;
  
  Blob<Dtype> aux_info_;
  shared_ptr<Blob<Dtype> > perturbed_labels_bias_;  // default for coordinates
};

// load the person masks, motion map or other imaegs as input
//    by GlobalVars -- imgidxs, objidx, and `aux_info` blob
//    which can corrspond to the input image data layer
// and then concat them with the output image data layer
template <typename Dtype>
class LoadImageDataLayer : public Layer<Dtype> {
 public:
  explicit LoadImageDataLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual inline const char* type() const { return "LoadImageData"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs()    const { return 1; }

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  /// @brief Not implemented -- 
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
  
 protected:
  virtual void load_data_image2blob(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  bool is_color_;
  bool is_rgb2gray_;
  std::string img_ext_;
  bool has_visual_path_;
  std::string root_folder_;
  std::string visual_path_;
};

/**
 * @brief During training only, sets a random portion of @f$x@f$ to 0, adjusting
 *        the rest of the vector magnitude accordingly.
 *
 * For a given convolution feature tensor of size (C, H, W), we perform only C dropout trails,
 * abd extebd the dropout value across the entire feature map. Therefore, adjacent pixels in the dropped-out
 * feature map are either all 0 (dropped-out) or all active.
 * We have found that this modified dropout (relative to standard dropout) implementation improves performace,
 * especially on the FLIC dataset, where the training set size is small.
 * Note that spatial-dropout happens before 1*1 convolution layer.
 *
 * Please refer to `Efficient Object Localization Using Convolutional Networks, CVPR 2015.`
 */
template <typename Dtype>
class SpatialDropoutLayer : public NeuronLayer<Dtype> {
 public:
  /**
   * @param param provides SpatialDropoutParameter SpatialDropout_param,
   *     with SpatialDropoutLayer options:
   *   - SpatialDropout_ratio (\b optional, default 0.5).
   *     Sets the probability @f$ p @f$ that any given unit is dropped.
   */
  explicit SpatialDropoutLayer(const LayerParameter& param)
      : NeuronLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "SpatialDropout"; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  /// when divided by UINT_MAX, the randomly generated values @f$u\sim U(0,1)@f$
  Blob<unsigned int> rand_vec_;
  /// the probability @f$ p @f$ of dropping any input
  Dtype threshold_;
  /// the scale for undropped inputs at train time @f$ 1 / (1 - p) @f$
  Dtype scale_;
  unsigned int uint_thres_;
};

/* the samce function as DataUpDownSamplesLayer */
template <typename Dtype>
class ResizeLayer : public Layer<Dtype> {
 public:
  explicit ResizeLayer(const LayerParameter& param)
  : Layer<Dtype>(param){}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
   virtual inline const char* type() const { 
    return "Resize"; 
  }
  // if bottom.size() == 1 -> please use ResizeParameter
  //    please refer to src/caffe/proto/caffe.proto for more details
  // if bottom.size() == 2 -> bottom[1]: (resize_width, resize_height)
  //    please refer to the layer which produces the bottom[1] blob for more details
  virtual inline int MinBottomBlobs() const { return 1; }
  virtual inline int MaxBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }
  // 只是简单的缩放
  virtual inline DiagonalAffineMap<Dtype> coord_map() {
    std::vector<std::pair<Dtype, Dtype> > coefs(2, make_pair(1, 0));
    coefs[0].first = scale_h_;
    coefs[1].first = scale_w_;
    return DiagonalAffineMap<Dtype>(coefs);
  }
  
 protected:
  virtual void Forward_cpu(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  virtual void Forward_gpu(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
 protected:
  bool has_bottom_param_;
  bool has_resize_param_;
  int out_height_;
  int out_width_;
  int out_channels_;
  int out_num_;
  Dtype scale_w_;
  Dtype scale_h_;
  vector<Blob<Dtype>*> locs_;
};

/**
 * @brief `heat map to coordinates` is that it creates coordinates from heat maps
 * simply find location of the maximum respondence in heat map and then according to
 * the mapping relationship between input image and the heat map, find the predicted
 * coordinate (x, y) of some part/joint
 *
 * Please refer to `Efficient Object Localization Using Convolutional Networks, CVPR 2015.`
 */
template <typename Dtype>
class HeatMapsFromCoordsLayer  : public Layer<Dtype> {
 public:
  explicit HeatMapsFromCoordsLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { 
    return "HeatMapsFromCoords"; 
  }
  // labels, aux info
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  // heat maps, mask, heat map info
  virtual inline int MinTopBlobs() const { return 2; }
  virtual inline int MaxTopBlobs() const { return 3; }
  
 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  /// @brief Not implemented -- 
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  // virtual void GetHeatMapHeightAndWidth(const vector<Blob<Dtype>*>& bottom);
  virtual void CreateHeatMapsFromCoords(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  
  bool is_binary_;
  int label_num_;
  int heat_map_a_;
  int heat_map_b_;
  int key_point_num_;
  int max_width_;
  int max_height_;
  int batch_num_;
  int heat_count_;
  int heat_num_;
  int heat_width_;
  int heat_height_;
  // heat_map_a_ * valid_dist_factor_
  float valid_dist_factor_;

  Blob<Dtype> prefetch_heat_maps_;  // 
  Blob<Dtype> prefetch_heat_maps_masks_;  // 
};

// Reference to: Efficient Object Localization Using Convolutional Networks, In CVPR 2015.
// Produce the heat map of each joint/part whose size is the same to the corresponding image
template <typename Dtype>
class HeatMapsFromCoordsOriginLayer  : public Layer<Dtype> {
 public:
  explicit HeatMapsFromCoordsOriginLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { 
    return "HeatMapsFromCoordsOrigin"; 
  }
  // labels, aux info
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  // heat maps, mask, heat map info
  virtual inline int MinTopBlobs() const { return 2; }
  virtual inline int MaxTopBlobs() const { return 3; }
  
 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  /// @brief Not implemented -- 
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  virtual void CreateHeatMapsFromCoordsOrigin(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  
  bool is_binary_;
  int radius_;
  int heat_num_;
  int batch_num_;
  int label_num_;
  int heat_width_;
  int heat_height_;
  int key_point_num_;
  float gau_mean_;
  float gau_stds_;
  std::vector<int> all_radius_;
  Blob<Dtype> prefetch_heat_maps_;
  Blob<Dtype> prefetch_heat_maps_masks_;
};

// Reference to: Zhujin.Liang iccv code
// Produce the heat map of each joint/part whose size is the same to the corresponding image
template <typename Dtype>
class HeatMapsFromCoordsOrigin2Layer  : public Layer<Dtype> {
 public:
  explicit HeatMapsFromCoordsOrigin2Layer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { 
    return "HeatMapsFromCoordsOrigin2"; 
  }
  // labels, aux info
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  // heat maps, mask, heat map info
  virtual inline int MinTopBlobs() const { return 2; }
  virtual inline int MaxTopBlobs() const { return 3; }
  
 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  /// @brief Not implemented -- 
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  virtual void Visualize();
  virtual void CreateHeatMapsFromCoordsOrigin2(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  
  bool is_binary_;
  bool has_visual_path_;
  int radius_;
  int in_radius_;
  int heat_num_;
  int batch_num_;
  int label_num_;
  int heat_width_;
  int heat_height_;
  int heat_channels_;
  int key_point_num_;
  float gau_mean_;
  float gau_stds_;
  std::string img_ext_;
  std::string visual_path_;
  std::string mask_visual_path_;
  std::string label_visual_path_;
  std::vector<int> all_radius_;
  std::vector<int> all_in_radius_;
  Blob<Dtype> prefetch_heat_maps_;
  Blob<Dtype> prefetch_heat_maps_masks_;
};

/**
 * @brief `heat map to coordinates` is that it creates coordinates from heat maps
 * simply find location of the maximum respondence in heat map and then according to
 * the mapping relationship between input image and the heat map, find the predicted
 * coordinate (x, y) of some part/joint
 *
 * bottom[0]: scale-origin coordinates (has been scaled, that means x' = x * im_scale)
 */
template <typename Dtype>
class CoordsFromHeatMapsLayer : public Layer<Dtype> {
 public:
  explicit CoordsFromHeatMapsLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { 
    return "CoordsFromHeatMaps"; 
  }
  // virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int MinBottomBlobs() const { return 1; }
  virtual inline int MaxBottomBlobs() const { return 2; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 2; }
  
 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  /// @brief Not implemented -- 
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void FilterHeatMap(const vector<Blob<Dtype>*>& bottom, 
        const vector<Blob<Dtype>*>& top);
  virtual void CreateCoordsFromHeatMap_cpu(const vector<Blob<Dtype>*>& bottom, 
      const vector<Blob<Dtype>*>& top);
  virtual void CreateCoordsFromHeatMapFromTopK_cpu(const vector<Blob<Dtype>*>& bottom, 
      const vector<Blob<Dtype>*>& top);

  // virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
  //     const vector<Blob<Dtype>*>& top);
  // /// @brief Not implemented -- 
  // virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
  //     const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  // void CreateCoordsFromHeatMap_gpu(const vector<Blob<Dtype>*>& bottom, 
  //     const vector<Blob<Dtype>*>& top);
  
  int topK_;
  int label_num_;
  int heat_map_a_;
  int heat_map_b_;
  int key_point_num_;
  int batch_num_;
  int heat_channels_;
  int heat_count_;
  int heat_num_;
  int heat_width_;
  int heat_height_;

  Blob<Dtype> prefetch_coordinates_scores_;  // 
  Blob<Dtype> prefetch_coordinates_labels_;  // default for coordinates
};

/**
 * @brief convert the coordinates of all parts to the whole bbox
 * bottom[0]: scale-origin coordinates (has been scaled, that means x' = x * im_scale)
 */
template <typename Dtype>
class CoordsToBboxesLayer : public Layer<Dtype>{
 public:
  explicit CoordsToBboxesLayer(const LayerParameter& param)
  : Layer<Dtype>(param){}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
   virtual inline const char* type() const { 
    return "CoordsToBboxes"; 
  }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
 protected:
  int bbox_id1_;
  int bbox_id2_;
  bool as_whole_;
};

// used to load one bbox of one image, like the torso
// use aux info to specify the `imgidxs` and `objidxs`
// 12
template <typename Dtype>
class LoadBboxFromFileLayer : public Layer<Dtype>{
 public:
  explicit LoadBboxFromFileLayer(const LayerParameter& param)
  : Layer<Dtype>(param){}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
   virtual inline const char* type() const { 
    return "LoadBboxFromFile"; 
  }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs()    const { return 1; }

 protected:
  virtual void Forward_cpu(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

 protected:
  bool is_color_;
  bool has_visual_path_;
  std::string img_ext_;
  std::string bbox_file_;
  std::string root_folder_;
  std::string visual_path_;
  // map: <imgidx, <objidx, coords>>
  std::map<std::string, std::map<int, std::vector<float> > > lines_;
};

/**
 * @brief convert the coordinates of all parts to the whole bbox
 * bottom[0]: scale-origin coordinates (has been scaled, that means x' = x * im_scale)
 * bottom[1]: aux_info
 */
template <typename Dtype>
class CoordsToBboxesMasksLayer : public Layer<Dtype>{
 public:
  explicit CoordsToBboxesMasksLayer(const LayerParameter& param)
  : Layer<Dtype>(param){}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
   virtual inline const char* type() const { 
    return "CoordsToBboxesMasks"; 
  }
  
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

  void InitRand();
  virtual int Rand(int n);

 protected:
  virtual void Forward_cpu(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
 
 protected:
  bool whole_;
  bool is_perturb_;
  bool has_visual_path_;
  int top_id_;
  int top_id2_;
  int top_idx_;
  int top_idx2_;
  int bottom_id_;
  int bottom_id2_;
  int bottom_idx_;
  int bottom_idx2_;
  int num_;
  int channels_;
  int n_channels_;
  int height_;
  int width_;
  int perb_num_;
  Dtype value_;
  std::string img_ext_;
  std::string visual_path_;
  shared_ptr<Caffe::RNG> rng_;
};

/**
 * @brief convert the coordinates of all parts to the whole bbox
 * bottom[0]: has been scaled, that means x' = x * im_scale
 *    scale-origin coordinates of all joints
 *  or
 *    the scale-bbox coordinates of only two joints which form a bbox
 * bottom[1]: aux_info
 * bottom[2]: some layer to resize the mask
 *  the mask maybe concat with intermediate layer, e.g. data layer, conv4 layer
 */
template <typename Dtype>
class TorsoMaskFromCoordsLayer : public Layer<Dtype>{
 public:
  explicit TorsoMaskFromCoordsLayer(const LayerParameter& param)
  : Layer<Dtype>(param){}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
   virtual inline const char* type() const { 
    return "TorsoMaskFromCoords"; 
  }
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs()    const { return 1; }
  virtual int Rand(int n);
  void InitRand();

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

 protected:
  bool whole_;
  bool has_input_path_;
  bool has_visual_path_;
  bool has_perb_num_;
  int perb_num_;
  int top_id_;
  int top_id2_;
  int top_idx_;
  int top_idx2_;
  int bottom_id_;
  int bottom_id2_;
  int bottom_idx_;
  int bottom_idx2_;
  int num_;
  int channels_;
  int n_channels_;
  int height_;
  int width_;
  Dtype value_;
  std::string img_ext_;
  std::string input_path_;
  std::string visual_path_;
  shared_ptr<Caffe::RNG> rng_;
};

/**
 * @brief Normalize the coordinates
 * bottom[0]: scale-origin coordinates (has been scaled, that means x' = x * im_scale)
 * bottom[1]: aux info (img_ind, ori-width, ori-height, im_scale)
 * top[0]: origin coordinates (need to be rescaled, that means x = x' / im_scale)
 */
template <typename Dtype>
class RescaledPoseCoordsLayer : public Layer<Dtype>{
 public:
  explicit RescaledPoseCoordsLayer(const LayerParameter& param)
  : Layer<Dtype>(param){}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
   virtual inline const char* type() const { 
    return "RescaledPoseCoords"; 
  }
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  
 protected:
  int num_;
  int channels_;
  int height_;
  int width_;
};

/**
 * @brief Normalize the coordinates
 * bottom[0]: scale-origin coordinates (has been scaled, that means x' = x * im_scale)
 * bottom[1]: aux info (img_ind, ori-width, ori-height, im_scale)
 */
template <typename Dtype>
class NormalizedPoseCoordsLayer : public Layer<Dtype>{
 public:
  explicit NormalizedPoseCoordsLayer(const LayerParameter& param)
  : Layer<Dtype>(param){}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
   virtual inline const char* type() const { 
    return "NormalizedPoseCoords"; 
  }
  virtual inline int MinBottomBlobs() const { return 1; }
  virtual inline int MaxBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  
 protected:
  bool has_statistic_file_;
 
  int num_;
  int channels_;
  int height_;
  int width_;
  // 0: do nothing
  // 1: deeppose -> x' = (x - half_width) / width, y' = (y - half_height) / height
  // 2: fashion parsing - > x' = (x - x_mean) / x_std, y' = (y - y_mean) / y_std
  int normalized_type_;
  std::string statistic_file_;

  Dtype real_width_;
  Dtype real_height_;

  std::vector<Dtype> coord_aves_;
  std::vector<Dtype> coord_stds_;
};

/**
 * @brief Revert-Normalize the coordinates
 * bottom[0]: scale-origin coordinates (has been scaled, that means x' = x * im_scale)
 * bottom[1]: aux info (img_ind, ori-width, ori-height, im_scale)
 */
template <typename Dtype>
class RevertNormalizedPoseCoordsLayer : public Layer<Dtype>{
 public:
  explicit RevertNormalizedPoseCoordsLayer(const LayerParameter& param)
  : Layer<Dtype>(param){}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
   virtual inline const char* type() const { 
    return "RevertNormalizedPoseCoords"; 
  }
  virtual inline int MinBottomBlobs() const { return 1; }
  virtual inline int MaxBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

 protected:
  bool has_statistic_file_;
 
  int num_;
  int channels_;
  int height_;
  int width_;
  // 0: do nothing
  // 1: deeppose -> x' = x * width + half_width, y' = y * height + half_height
  // 2: fashion parsing - > x' = x * x_std + x_mean, y' = y * y_std + y_mean
  int normalized_type_;
  std::string statistic_file_;

  Dtype real_width_;
  Dtype real_height_;

  std::vector<Dtype> coord_aves_;
  std::vector<Dtype> coord_stds_;
};

// if bottom.size() == 1:
//  bottom[0]: either predicted or ground truth
// if bottom.size() == 2:
//  bottom[0]: predicted
//  bottom[1]: ground truth
// Visualize Heat maps for both predicted and ground truth 
template <typename Dtype>
class VisualizedHeatMapsLayer : public Layer<Dtype>{
 public:
  explicit VisualizedHeatMapsLayer(const LayerParameter& param)
  : Layer<Dtype>(param){}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
   virtual inline const char* type() const { 
    return "VisualizedHeatMaps"; 
  }
  virtual inline int MinBottomBlobs() const { return 1; }
  virtual inline int MaxBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 0; }

 protected:
  virtual void Forward_cpu(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  void WriteFiles(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  void WriteImages(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
 protected:
  int num_;
  int channels_;
  int height_;
  int width_;
  int heat_num_;
  // 0: file-format, 
  // 1: image-format
  // 2: both
  int visual_type_;
  Dtype threshold_;
  // path
  std::string heat_map_path_;
  std::string heat_map_files_name_;
  std::string heat_map_files_path_;
  std::string heat_map_images_name_;
  std::string heat_map_images_path_;
  std::string phase_name_;
  std::string phase_path_;
  std::string img_ext_;
  std::string file_ext_;

  // come from caffe/util/global_variables.hpp
  std::vector<std::string> objidxs_;
  std::vector<std::string> imgidxs_;
  std::vector<std::string> images_paths_;
};

// if bottom.size() == 2:
//  bottom[0]: either predicted or ground truth
//  bottom[1]: aux_info
// if bottom.size() == 3:
//  bottom[0]: predicted
//  bottom[1]: ground truth
//  bottom[2]: aux_info
// Visualize Heat maps for both predicted and ground truth
// But predict ground truth have the same size
template <typename Dtype>
class VisualizedHeatMaps2Layer : public Layer<Dtype>{
 public:
  explicit VisualizedHeatMaps2Layer(const LayerParameter& param)
  : Layer<Dtype>(param){}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
   virtual inline const char* type() const { 
    return "VisualizedHeatMaps2"; 
  }
  virtual inline int MinBottomBlobs() const { return 2; }
  virtual inline int MaxBottomBlobs() const { return 3; }
  virtual inline int ExactNumTopBlobs() const { return 0; }

 protected:
  virtual void Forward_cpu(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  void WriteFiles(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  void WriteImages(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
 protected:
  int num_;
  int channels_;
  int height_;
  int width_;
  int heat_num_;
  int max_width_;
  int max_height_;
  // 0: file-format, 
  // 1: image-format
  // 2: both
  int visual_type_;
  Dtype threshold_;
  // path
  std::string heat_map_path_;
  std::string heat_map_files_name_;
  std::string heat_map_files_path_;
  std::string heat_map_images_name_;
  std::string heat_map_images_path_;
  std::string phase_name_;
  std::string phase_path_;
  std::string img_ext_;
  std::string file_ext_;

  // come from caffe/util/global_variables.hpp
  std::vector<std::string> objidxs_;
  std::vector<std::string> imgidxs_;
  std::vector<std::string> images_paths_;
};

// pre-condition: all coors (blobs) must in the same origin-scale
template <typename Dtype>
class VisualizedPoseCoordsLayer : public Layer<Dtype>{
 public:
  explicit VisualizedPoseCoordsLayer(const LayerParameter& param)
  : Layer<Dtype>(param){}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
   virtual inline const char* type() const { 
    return "VisualizedPoseCoords"; 
  }
  // predicted/ground-truth, aux info
  virtual inline int MinBottomBlobs() const { return 2; }
  // predicted, ground-truth, aux info
  virtual inline int MaxBottomBlobs() const { return 3; }
  virtual inline int ExactNumTopBlobs() const { return 0; }

 protected:
  virtual void Forward_cpu(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

 void WriteFiles(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  void WriteImages(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
 protected:
  bool is_draw_skel_;
  bool is_draw_text_;
  int num_;
  int channels_;
  int height_;
  int width_;
  int key_point_num_;
  std::vector<int> start_skel_idxs_;
  std::vector<int> end_skel_idxs_;
  std::string coords_path_;
  std::string coords_files_name_;
  std::string coords_files_path_;
  std::string coords_images_name_;
  std::string coords_images_path_;
  std::vector<std::string> phase_name_;
  std::string skel_path_;
  std::string img_ext_;
  std::string file_ext_;
  // come from caffe/util/global_variables.hpp
  std::vector<std::string> objidxs_;
  std::vector<std::string> imgidxs_;
  std::vector<std::string> images_paths_;
};

// pre-condition: need bottom[bottom.size() -1] blob to rescale the coordinates
template <typename Dtype>
class VisualizedPoseCoords2Layer : public Layer<Dtype>{
 public:
  explicit VisualizedPoseCoords2Layer(const LayerParameter& param)
  : Layer<Dtype>(param){}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
   virtual inline const char* type() const { 
    return "VisualizedPoseCoords2"; 
  }
  // predicted/ground-truth, aux info
  virtual inline int MinBottomBlobs() const { return 2; }
  // predicted, ground-truth, aux info
  virtual inline int MaxBottomBlobs() const { return 3; }
  virtual inline int ExactNumTopBlobs() const { return 0; }

 protected:
  virtual void Forward_cpu(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

 void WriteFiles(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  void WriteImages(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
 protected:
  bool is_draw_skel_;
  bool is_draw_text_;
  int num_;
  int channels_;
  int height_;
  int width_;
  int key_point_num_;
  std::vector<int> start_skel_idxs_;
  std::vector<int> end_skel_idxs_;
  std::string coords_path_;
  std::string coords_files_name_;
  std::string coords_files_path_;
  std::string coords_images_name_;
  std::string coords_images_path_;
  std::vector<std::string> phase_name_;
  std::string skel_path_;
  std::string img_ext_;
  std::string file_ext_;
  // come from caffe/util/global_variables.hpp
  std::vector<std::string> objidxs_;
  std::vector<std::string> imgidxs_;
  std::vector<std::string> images_paths_;
};

// data/images: multi-sources for per image, 
//  like <color, optical flow, ...> or <color, depth, ...>
// by using the same `imgidx`
// labels/coords
// aux info (img_ind, width, height, im_scale)
// support multi-source and multi-scale
template <typename Dtype>
class MultiSourcesImagesDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit MultiSourcesImagesDataLayer(
    const LayerParameter& param)
      : BasePrefetchingDataLayer<Dtype>(param) {}
  virtual ~MultiSourcesImagesDataLayer();
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { 
    return "MultiSourcesImagesData"; 
  }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 3; }

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

 protected:
  shared_ptr<Caffe::RNG> prefetch_rng_;
  void PerturbedCoordsBias();
  virtual void ShuffleImages();
  virtual void InternalThreadEntry();

  bool shuffle_;
  bool is_scale_image_;
  bool always_max_size_;
  bool has_mean_values_;
  bool is_perturb_coords_;

  int lines_id_;
  int max_size_;
  int label_num_;
  int crop_size_;
  int n_sources_;
  int batch_size_;
  int key_points_num_;
  int receptive_field_size_;
  
  float min_plb_;
  float max_plb_;
  
  vector<bool> is_colors_;
  std::vector<int> min_sizes_;
  std::vector<int> origin_parts_orders_;
  std::vector<int> flippable_parts_orders_;


  std::string parts_orders_path_;

  std::vector<string> im_exts_;
  std::vector<std::string> sources_;
  std::vector<std::string> root_folders_;

  // variables -- global
  std::vector<std::string> objidxs_;
  std::vector<std::string> imgidxs_;
  std::vector<std::string> images_paths_;

  std::vector<int> inds_;
  std::vector<int> channels_inds_;
  vector<Dtype> mean_values_;
  // <imgidx, <objidx, coords>>
  vector<std::pair<int, std::pair<std::string, std::pair<std::string, 
                        std::vector<float> > > > >lines_;
  
  
  Blob<Dtype> aux_info_;
  // default for coordinates
  shared_ptr<Blob<Dtype> > perturbed_labels_bias_;  
};


}  // namespace caffe

#endif  // CAFFE_POSE_ESTIMATION_LAYERS_HPP_