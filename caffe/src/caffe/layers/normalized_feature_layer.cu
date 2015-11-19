#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/fast_rcnn_action_layers.hpp"

using namespace std;

namespace caffe {


template <typename Dtype>
void NormalizeFeatureLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) 
{
  if(norm_feat_param_.norm_method() == NormalizeFeaturesParameter_NormMethod_L1Sq) {
    // Get data pointers
    const Dtype* bottom_data = bottom[0]->gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data();
    CHECK_EQ(bottom[0]->count(), top[0]->count());
    const int count = bottom[0]->count();

    if(norm_feat_param_.feat_split_method() == 
        NormalizeFeaturesParameter_FeatSplitMethod_Whole) 
    {
      CHECK_EQ(fdot_sum_.count(), 1);
      CHECK_EQ(fsqrt_sum_.count(), 1);
      
      // Calculate the sum of square of per element, and sqrt
      Dtype dot;
      caffe_gpu_dot(count, bottom_data, bottom_data, &dot);
      fdot_sum_.mutable_cpu_data()[0] = dot;
      const Dtype sq = std::sqrt(dot);
      fsqrt_sum_.mutable_cpu_data()[0] = sq;
      const Dtype sq_scaler = Dtype(1.) / sq;
      // Normalize
      caffe_copy(count, bottom_data, top_data);
      caffe_gpu_scal(count, sq_scaler, top_data);
    }

    // ##############################################################################
    
    if(norm_feat_param_.feat_split_method() == 
          NormalizeFeaturesParameter_FeatSplitMethod_Batch) 
    {
      CHECK_EQ(fdot_sum_.count(), num_);
      CHECK_EQ(fsqrt_sum_.count(), num_);

      // Start
      int offset = 0;
      const int sub_count = count / num_;
      for(int n = 0; n < num_; n++) {
        // Calculate the sum of square of per element, and sqrt
        Dtype dot;
        caffe_gpu_dot(sub_count, bottom_data + offset, bottom_data + offset, &dot);
        fdot_sum_.mutable_cpu_data()[n] = dot;
        const Dtype sq = std::sqrt(dot);
        fsqrt_sum_.mutable_cpu_data()[n] = sq;
        const Dtype sq_scaler = Dtype(1.) / sq;
        // Normalize
        caffe_copy(sub_count, bottom_data + offset, top_data + offset);
        caffe_gpu_scal(sub_count, sq_scaler, top_data + offset);
        // Add offset
        offset += sub_count;
      }
      CHECK_EQ(offset, count);
    }

    // Blob<Dtype> out;
    // out.Reshape(top[0]->num(), top[0]->channels(), top[0]->height(), top[0]->width());
    // caffe_copy(top[0]->count(), top_data, out.mutable_cpu_data());
    // const int out_len = 20;
    // for(int ol = 0; ol < out_len; ol++) {
    //   LOG(INFO) << out.cpu_data()[ol];
    // }
  }
  else {
    LOG(INFO) << "L1 or L2 normalization has not been implemented yet...";
    NOT_IMPLEMENTED;
  }


}

template <typename Dtype>
void NormalizeFeatureLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) 
{  
  if(!propagate_down[0]) return;

  if(norm_feat_param_.norm_method() == NormalizeFeaturesParameter_NormMethod_L1Sq) {
    // Get data pointers
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int count = bottom[0]->count();

    caffe_copy(multiplier_.count(), multiplier_.cpu_data(), multiplier_.mutable_gpu_data());
    if(norm_feat_param_.feat_split_method() == 
        NormalizeFeaturesParameter_FeatSplitMethod_Whole) 
    {
      Dtype sum;
      caffe_gpu_dot(count, multiplier_.gpu_data(), bottom_data, &sum);
      const Dtype eof1 = sum / (fdot_sum_.cpu_data()[0] * fsqrt_sum_.cpu_data()[0]);
      caffe_copy(count, bottom_data, bottom_diff);
      caffe_gpu_scal(count, -eof1, bottom_diff);
      const Dtype eof2 = Dtype(1) / fsqrt_sum_.cpu_data()[0];
      caffe_gpu_add_scalar(count, eof2, bottom_diff);
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
        Dtype sum;
        caffe_gpu_dot(sub_count, multiplier_.gpu_data(), bottom_data + offset, &sum);
        const Dtype eof1 = sum / (fdot_sum_.cpu_data()[n] * fsqrt_sum_.cpu_data()[n]);
        caffe_copy(sub_count, bottom_data + offset, bottom_diff + offset);
        caffe_gpu_scal(sub_count, -eof1, bottom_diff + offset);
        const Dtype eof2 = Dtype(1) / fsqrt_sum_.cpu_data()[n];
        caffe_gpu_add_scalar(sub_count, eof2, bottom_diff + offset);
        // Add offset
        offset += sub_count;
      }
    }

    // finally multiply the residual from previous layer
    caffe_gpu_mul(count, top_diff, bottom[0]->gpu_diff(), bottom[0]->mutable_gpu_diff());
  } else {
    LOG(INFO) << "L1 or L2 normalization has not been implemented yet...";
    NOT_IMPLEMENTED;
  }
}


INSTANTIATE_LAYER_GPU_FUNCS(NormalizeFeatureLayer);


}  // namespace caffe
