// Copyright 2015 Zhu.Jin Liang

#include <algorithm>
#include <map>
#include <set>
#include <vector>
#include <iostream>

#include "caffe/layer.hpp"
#include "caffe/common.hpp"

#include "caffe/zhu_face_layers.hpp"
#include "caffe/util/util_img.hpp"
#include "caffe/util/util_coords.hpp"
#include "caffe/util/util_pre_define.hpp"

namespace caffe {

template <typename Dtype>
void ResizeWithMapLayer<Dtype>::Forward_gpu(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

	for (int i = 0; i < top.size(); ++i) {
		AffineWarpBlob_gpu(bottom[i + 2], top[i], &weights_, &locs_);
	}
}

template <typename Dtype>
__global__ void ResizeWithMapBackward(
		const int nthreads,
		Dtype* bottom_diff, 
		const int bottom_step,
		const Dtype* top_diff, 
		const int top_step, 
		const int top_channels,
		const Dtype* weights, 
		const int* locs,
		const int weight_num, 
		const int weight_channels,
		const int weight_height, 
		const int weight_width) 
{
  CUDA_KERNEL_LOOP(index, nthreads) {

  	// 把num跟channels合成一个维度
  	// 算出当前是第几个height * width
  	// 以及在该height * width下的位置
  	const int i = index / top_step;
  	const int j = index % top_step;

  	// 算出是第几个样本
  	const int n = i / top_channels;
  	// 求出是第几个channels
  	const int c = i % top_channels;
  	// 算出该channels下对应的x/y映射的offset
  	const int weight_n = n % weight_num;
  	const int weight_c = c / (top_channels / weight_channels);

  	const int in_offset = i * bottom_step;
  	const int weights_offset = ((weight_n * weight_channels + weight_c) * weight_height + j) * weight_width;

  	for (int k = 0; k < weight_width; ++k) {
  		bottom_diff[in_offset + locs[weights_offset + k]] += (weights[weights_offset + k] * top_diff[index]);
  	}
  }
}

template <typename Dtype>
void ResizeWithMapLayer<Dtype>::Backward_gpu(
		const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) 
{
  const Dtype* weights = weights_.gpu_data();
  const int* locs = locs_.gpu_data();

	for (int i = 0; i < top.size(); ++i) {
		if (propagate_down[i + 2]) {

		  Dtype* bottom_diff = bottom[i + 2]->mutable_gpu_diff();
			const int bottom_step = bottom[i + 2]->height() * bottom[i + 2]->width();
			caffe::caffe_gpu_set(bottom[i + 2]->count(), Dtype(0.), bottom_diff);

		  const Dtype* top_diff = top[i]->gpu_diff();
			const int top_step = top[i]->height() * top[i]->width();

			const int count = top[i]->count();

			ResizeWithMapBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
					count,
					bottom_diff, bottom_step,
					top_diff, top_step, top[i]->channels(),
					weights, locs,
					weights_.num(), weights_.channels(),
					weights_.height(), weights_.width());
			CUDA_POST_KERNEL_CHECK;
		}
	}
}

INSTANTIATE_LAYER_GPU_FUNCS(ResizeWithMapLayer);

}  // namespace caffe
