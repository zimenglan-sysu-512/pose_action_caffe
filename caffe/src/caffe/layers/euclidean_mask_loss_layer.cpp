// Copyright 2015 Zhu.Jin Liang

#include <algorithm>
#include <cmath>
#include <cfloat>
#include <vector>
#include <sstream>

#include "caffe/layer.hpp"
#include "caffe/common.hpp"

#include "caffe/zhu_face_layers.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/util_pre_define.hpp"

using std::max;

namespace caffe {

template<typename Dtype>
void EuclideanMaskLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	const int count = bottom[0]->count();
	Dtype* diff_data = this->diff_.mutable_cpu_data();
	memset(diff_data, 0, sizeof(Dtype) * this->diff_.count());
	const Dtype* target = bottom[1]->cpu_data();
	const int mask_length = this->layer_param_.euclidean_mask_param().mask_length();
	// only update if there are bboxes in target
	for (int i = 0; i < count; i += mask_length) {
		int j = 0;
		for (; j < mask_length; ++j) {
			if (IsValidCoord(target[i + j])) break;
		}
		if (j < mask_length) {
			caffe_sub(mask_length, bottom[0]->cpu_data() + i, bottom[1]->cpu_data() + i,
					diff_data + i);
		}
	}
	//	caffe_sub(count, bottom[0]->cpu_data(), bottom[1]->cpu_data(),
	//			this->diff_.mutable_cpu_data());
	Dtype dot = caffe_cpu_dot(count, this->diff_.cpu_data(), this->diff_.cpu_data());
  Dtype loss = dot / bottom[0]->num() / Dtype(2);

  top[0]->mutable_cpu_data()[0] = loss;

	//	LOG(INFO) << "Loss = " << loss << ", Sample counts: " << actual_num;
}

#ifdef CPU_ONLY
STUB_GPU(EuclideanMaskLossLayer);
#endif

INSTANTIATE_CLASS(EuclideanMaskLossLayer);
REGISTER_LAYER_CLASS(EuclideanMaskLoss);

}  // namespace caffe