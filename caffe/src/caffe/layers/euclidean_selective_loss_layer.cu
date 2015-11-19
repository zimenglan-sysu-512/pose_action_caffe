// Copyright 2015 Zhu.Jin Liang

#include <vector>

#include "caffe/layer.hpp"
#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"

#include "caffe/zhu_face_layers.hpp"
#include "caffe/util/util_pre_define.hpp"

namespace caffe {

template <typename Dtype>
void EuclideanSelectiveLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	// TODO
	return Forward_cpu(bottom, top);
}

template <typename Dtype>
void EuclideanSelectiveLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
      caffe_gpu_axpby(
          bottom[i]->count(),                 // count
          alpha,                              // alpha
          diff_.gpu_data(),                   // a
          Dtype(0),                           // beta
          bottom[i]->mutable_gpu_diff());     // b
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(EuclideanSelectiveLossLayer);

}  // namespace caffe
