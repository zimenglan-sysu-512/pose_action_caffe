// Copyright 2015 BVLC and contributors.

#include <vector>
#include <sstream>
#include <iostream>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/global_variables.hpp"
#include "caffe/wanglan_face_shoulders_layers.hpp"

namespace caffe {

template <typename Dtype>
void EuclideanLoss2Layer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
      << "Inputs must have the same dimension.";
  diff_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void EuclideanLoss2Layer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) 
{
  int count = bottom[0]->count();
  /// compute gradient
  caffe_sub(
      count,
      bottom[0]->cpu_data(),
      bottom[1]->cpu_data(),
      diff_.mutable_cpu_data());
  Dtype dot = caffe_cpu_dot(count, diff_.cpu_data(), diff_.cpu_data());
  Dtype loss = dot / bottom[0]->num() / Dtype(2);
  /// print predicted info
  if((getenv("PREDICT_PRINT") != NULL) && 
      (getenv("PREDICT_PRINT")[0] == '1')) 
  {
    LOG(INFO) << "prediction vs groundtruth: ";
    for(int c = 0; c < count; c++) {
      LOG(INFO) << c << ": " << bottom[0]->cpu_data()[c] 
          << "\t" << bottom[1]->cpu_data()[c];
    }
  }
  /// print loss info
  if (getenv("LOSS_PRINT") != NULL) {
    char choice = getenv("LOSS_PRINT")[0];
    if(choice == '1' || choice == '2') {
      int dims = count / bottom[0]->num();
      if(choice == '2') {
        LOG(INFO) << "loss: ";
        for(int n = 0; n < bottom[0]->num(); n++) {
          Dtype tmp_loss = caffe_cpu_dot(
              dims, 
              diff_.cpu_data() + n * dims, 
              diff_.cpu_data() + n * dims);
          LOG(INFO) << n << ": " << tmp_loss;
        }
        LOG(INFO);
      }
      LOG(INFO) << "L2 loss: " << loss;
      //
      Dtype patch_loss = 0;
      for(int c = 0; c < bottom[0]->count(); ++c) {
        if (bottom[1]->cpu_data()[c] != 0) {
          patch_loss += (diff_.cpu_data()[c] * diff_.cpu_data()[c]);
        }
      }
      patch_loss = patch_loss / bottom[0]->num() / Dtype(2);
      LOG(INFO) << "Patch loss: " << patch_loss;
    } // end inner if
  } // end outer if
  /// set loss
  LOG(INFO) << "Iteration: " << GlobalVars::caffe_iter();
  LOG(INFO) << "Learn Rate: " << GlobalVars::learn_lr();
  LOG(INFO) << "Euclidean Loss: " << loss;
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void EuclideanLoss2Layer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) 
{
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
      caffe_cpu_axpby(
          bottom[i]->count(),              // count
          alpha,                              // alpha
          diff_.cpu_data(),                   // a
          Dtype(0),                           // beta
          bottom[i]->mutable_cpu_diff());  // b
    }
    /// print gradient
    if ((getenv("DIFF_PRINT") != NULL) && (getenv("DIFF_PRINT")[0] == '1')) {
      LOG(INFO) << "gradient: ";
      for(int c = 0; c < bottom[i]->count(); ++c) {
        LOG(INFO) << c << ": " << bottom[i]->cpu_diff()[c];
      }
      LOG(INFO);
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(EuclideanLoss2Layer);
#endif

INSTANTIATE_CLASS(EuclideanLoss2Layer);
REGISTER_LAYER_CLASS(EuclideanLoss2);

}  // namespace caffe
