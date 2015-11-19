#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/fast_rcnn_action_layers.hpp"

using namespace std;

namespace caffe {

template <typename Dtype>
void NormalizeFeatureLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) 
{
  CHECK(this->layer_param_.has_norm_feat_param());
  norm_feat_param_ = this->layer_param_.norm_feat_param();

  CHECK(norm_feat_param_.has_norm_method());
  CHECK(norm_feat_param_.has_feat_split_method());

  if(norm_feat_param_.norm_method() == NormalizeFeaturesParameter_NormMethod_L1 ||
      norm_feat_param_.norm_method() == NormalizeFeaturesParameter_NormMethod_L2) {
    LOG(INFO) << "L1 or L2 normalization has not been implemented yet...";
    NOT_IMPLEMENTED;
  }
}

/*
  We either regard each (1, channels, height, width) as a feature, 
  or regard (num, channels, height, width) as a feature
  Normalizing among these features is independent.
*/
template <typename Dtype>
void NormalizeFeatureLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) 
{
  num_ = bottom[0]->num();
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();

  if(norm_feat_param_.feat_split_method() == 
      NormalizeFeaturesParameter_FeatSplitMethod_Whole) {
    fdot_sum_.Reshape(1, 1, 1, 1);
    fsqrt_sum_.Reshape(1, 1, 1, 1);
    // 
    multiplier_.Reshape(num_, channels_, height_, width_);
    caffe_set(multiplier_.count(), Dtype(1),
        multiplier_.mutable_cpu_data());
  } else if(norm_feat_param_.feat_split_method() == 
      NormalizeFeaturesParameter_FeatSplitMethod_Batch) {
    fdot_sum_.Reshape(num_, 1, 1, 1);
    fsqrt_sum_.Reshape(num_, 1, 1, 1);
    // 
    multiplier_.Reshape(1, channels_, height_, width_);
    caffe_set(multiplier_.count(), Dtype(1),
        multiplier_.mutable_cpu_data());
  } else {
    NOT_IMPLEMENTED;
  }

  // Reshape the top blob
  top[0]->Reshape(num_, channels_, height_, width_);
}

template <typename Dtype>
void NormalizeFeatureLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) 
{
  if(norm_feat_param_.norm_method() == NormalizeFeaturesParameter_NormMethod_L1Sq) {
    // Get data pointers
    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* top_data = top[0]->mutable_cpu_data();
    CHECK_EQ(bottom[0]->count(), top[0]->count());
    const int count = bottom[0]->count();

    if(norm_feat_param_.feat_split_method() == 
        NormalizeFeaturesParameter_FeatSplitMethod_Whole) 
    {
      CHECK_EQ(fdot_sum_.count(), 1);
      CHECK_EQ(fsqrt_sum_.count(), 1);
      
      // Calculate the sum of square of per element, and sqrt
      const Dtype dot = caffe_cpu_dot(count, bottom_data, bottom_data);
      fdot_sum_.mutable_cpu_data()[0] = dot;
      const Dtype sq = std::sqrt(dot);
      fsqrt_sum_.mutable_cpu_data()[0] = sq;
      const Dtype sq_scaler = Dtype(1.) / sq;
      // Normalize
      caffe_copy(count, bottom_data, top_data);
      caffe_scal(count, sq_scaler, top_data);
    }

    // ##############################################################################
    
    if(norm_feat_param_.feat_split_method() == 
          NormalizeFeaturesParameter_FeatSplitMethod_Batch) 
    {
      CHECK_EQ(fdot_sum_.count(), num_);
      CHECK_EQ(fsqrt_sum_.count(), num_);

      // Start
      int offset = 0;
      const int count = bottom[0]->count();
      const int sub_count = count / num_;
      for(int n = 0; n < num_; n++) {
        // Calculate the sum of square of per element, and sqrt
        const Dtype dot = caffe_cpu_dot(sub_count,  bottom_data + offset, bottom_data + offset);
        fdot_sum_.mutable_cpu_data()[n] = dot;
        const Dtype sq = std::sqrt(dot);
        fsqrt_sum_.mutable_cpu_data()[n] = sq;
        const Dtype sq_scaler = Dtype(1.) / sq;
        // Normalize
        caffe_copy(sub_count, bottom_data + offset, top_data + offset);
        caffe_scal(sub_count, sq_scaler, top_data + offset);
        // Add offset
        offset += sub_count;
      }
      CHECK_EQ(offset, count);
    }
  }
  else {
    LOG(INFO) << "L1 or L2 normalization has not been implemented yet...";
    NOT_IMPLEMENTED;
  }


}

template <typename Dtype>
void NormalizeFeatureLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) 
{  
  if(!propagate_down[0]) return;

  if(norm_feat_param_.norm_method() == NormalizeFeaturesParameter_NormMethod_L1Sq) {
    // f(xi)` = ((x1^2 + x2^2 + ... + xn^2) - xi * (x1 + x2 + ... + xn)) / ((x1^2 + x2^2 + ... + xn^2)^(3/2))
    // f(xi)` = - xi * (x1 + x2 + ... + xn) / ((x1^2 + x2^2 + ... + xn^2)^(3/2)) + (x1^2 + x2^2 + ... + xn^2) / ((x1^2 + x2^2 + ... + xn^2)^(3/2)) 
    // f(xi)` = - xi * (x1 + x2 + ... + xn) / ((x1^2 + x2^2 + ... + xn^2)^(3/2)) + 1 / ((x1^2 + x2^2 + ... + xn^2)^(1/2)) 
    // f(xi)` = (- sum / (fdot_sum_[0] * fsqrt_sum_[0])) * xi + 1 / fsqrt_sum_[0]
    // Get data pointers
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();

    if(norm_feat_param_.feat_split_method() == 
        NormalizeFeaturesParameter_FeatSplitMethod_Whole) 
    {
      const Dtype sum = caffe_cpu_dot(count, multiplier_.cpu_data(), bottom_data);
      const Dtype eof1 = sum / (fdot_sum_.cpu_data()[0] * fsqrt_sum_.cpu_data()[0]);
      caffe_copy(count, bottom_data, bottom_diff);
      caffe_scal(count, -eof1, bottom_diff);
      const Dtype eof2 = Dtype(1) / fsqrt_sum_.cpu_data()[0];
      caffe_add_scalar(count, eof2, bottom_diff);
    }

    // ##############################################################################
    
    if(norm_feat_param_.feat_split_method() == 
          NormalizeFeaturesParameter_FeatSplitMethod_Batch) 
    {
      // Start
      int offset = 0;
      const int sub_count = count / num_;
      // Start
      for(int n = 0; n < num_; n++) {
        const Dtype sum = caffe_cpu_dot(sub_count, multiplier_.cpu_data(), bottom_data + offset);
        const Dtype eof1 = sum / (fdot_sum_.cpu_data()[n] * fsqrt_sum_.cpu_data()[n]);
        caffe_copy(sub_count, bottom_data + offset, bottom_diff + offset);
        caffe_scal(sub_count, -eof1, bottom_diff + offset);
        const Dtype eof2 = Dtype(1) / fsqrt_sum_.cpu_data()[n];
        caffe_add_scalar(sub_count, eof2, bottom_diff + offset);
        // Add offset
        offset += sub_count;
      }
      CHECK_EQ(offset, count);
    }

    // finally multiply the residual from previous layer
    caffe_mul(count, top_diff, bottom[0]->cpu_diff(), bottom_diff);
  } else {
    LOG(INFO) << "L1 or L2 normalization has not been implemented yet...";
    NOT_IMPLEMENTED;
  }
}


#ifdef CPU_ONLY
STUB_GPU(NormalizeFeatureLayer);
#endif

INSTANTIATE_CLASS(NormalizeFeatureLayer);
REGISTER_LAYER_CLASS(NormalizeFeature);

}  // namespace caffe
