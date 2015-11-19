// Copyright 2015 Zhu.Jin Liang

#include <vector>

#include "caffe/layer.hpp"
#include "caffe/common.hpp"

#include "caffe/zhu_face_layers.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/util_pre_define.hpp"


namespace caffe {

template <typename Dtype>
__global__ void EuclideanMaskLossForward(int nthreads, const Dtype* bottom_data,
		const Dtype* target_data, Dtype* diff, const int mask_length) {
	CUDA_KERNEL_LOOP(index, nthreads) {
		int i = index * mask_length;
		int j = 0;
		for (; j < mask_length; ++j) {
			diff[i + j] = 0;
		}
		for (j = 0; j < mask_length; ++j) {
			if (IsValidCoord(target_data[i + j]) > ELLISION) break;
		}
		if (j < mask_length) {
			for (j = 0; j < mask_length; ++j) {
				diff[i + j] = bottom_data[i + j] - target_data[i + j];
			}
		}
	}
}

template <typename Dtype>
void EuclideanMaskLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	const Dtype* bottom_data = bottom[0]->gpu_data();
	const Dtype* target_data = bottom[1]->mutable_gpu_data();
	Dtype* diff_data = this->diff_.mutable_gpu_data();
  const int mask_length =
  		this->layer_param_.euclidean_mask_param().mask_length() > 0 ?
  				this->layer_param_.euclidean_mask_param().mask_length() : 1;
  const int count = bottom[0]->count() / mask_length;
  EuclideanMaskLossForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
          count, bottom_data, target_data, diff_data, mask_length);
  CUDA_POST_KERNEL_CHECK;

  Dtype dot;
  caffe_gpu_dot(this->diff_.count(), this->diff_.gpu_data(), this->diff_.gpu_data(), &dot);
  Dtype loss = dot / bottom[0]->num() / Dtype(2);

  top[0]->mutable_cpu_data()[0] = loss;
}

INSTANTIATE_LAYER_GPU_FORWARD(EuclideanMaskLossLayer);

}  // namespace caffe
