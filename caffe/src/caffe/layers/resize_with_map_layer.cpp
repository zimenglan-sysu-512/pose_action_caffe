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
void ResizeWithMapLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
	first_info_ = false;
}

template <typename Dtype>
void ResizeWithMapLayer<Dtype>::Reshape(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
	CHECK_EQ(bottom.size() - 2, top.size());
	CHECK_EQ(bottom[0]->num(), bottom[1]->num());

	for (int i = 2; i < bottom.size(); ++i) {
		CHECK_EQ(bottom[0]->num(), bottom[i]->num());
		CHECK_EQ(bottom[0]->width(), bottom[i]->width())
				<< "The width of bottom[2..n] shoud be equal to bottom[0]";
		CHECK_EQ(bottom[0]->height(), bottom[i]->height())
				<< "The height of bottom[2..n] shoud be equal to bottom[0]";
	}

	coefs_ = GetMapBetweenFeatureMap(bottom, this->net_).inv().coefs();
	for (int i = 0; i < coefs_.size() && !first_info_; ++i) {
		LOG(INFO) << this->layer_param_.name() << ": " << coefs_[i].first << ": " << coefs_[i].second;
	}
	first_info_ = true;

	CHECK_GE(coefs_.size(), 2)
			<< ": The size of coefs( " << coefs_.size() << ") should greater than 2";;
	CHECK(coefs_.size() % 2 == 0)
			<< ": The size of coefs( " << coefs_.size() << ") should be even";

	coefs_mode_ = SINGLE_SAMPLE_SINGLE_CHANNEL;
	// 为了能够处理多patch的resize
	if (coefs_.size() > 2) {
		const int coefs_count = coefs_.size() / 2;

		bool isOK = true;
		int failed_i;
		// 如果是多patch的情况，那么有可能是不同样本的coefs不一样
		// 这时候需要
		//			coefs能被样本数整除，且coefs除以样本数得到的商能整除channels数
		coefs_mode_ = MULTI_SAMPLE_MULTI_CHANNEL;
		for (failed_i = 2; failed_i < bottom.size() && isOK; ++failed_i) {
			isOK = (coefs_count % bottom[failed_i]->num() == 0);
			int q = coefs_count / bottom[failed_i]->num();
			isOK = (isOK && (bottom[failed_i]->channels() % q == 0));
		}
		if (!isOK) {
			isOK = true;
			coefs_mode_ = SINGLE_SAMPLE_MULTI_CHANNEL;
			// 如果不是多patch的情况，则要严格要求bottom[i]的channels能被coefs的size整除
			for (failed_i = 2; failed_i < bottom.size() && isOK; ++failed_i) {
				isOK = (bottom[failed_i]->channels() % coefs_count == 0);
			}
		}

		CHECK(isOK) << bottom[failed_i - 1]->shape_string()
				<< ", size of coefs: " <<  coefs_.size();

	}

	for (int i = 0; i < top.size(); ++i) {
		top[i]->Reshape(bottom[i + 2]->num(), bottom[i + 2]->channels(), bottom[1]->height(),
				bottom[1]->width());
	}

	// get resize rules
	const int num = (coefs_mode_ == MULTI_SAMPLE_MULTI_CHANNEL ? bottom[0]->num() : 1);
	const int coord_maps_count = coefs_.size() / num / 2;
	GetResizeRules(bottom[0]->height(), bottom[0]->width(),
			bottom[1]->height(), bottom[1]->width(),
			&weights_, &locs_, coefs_, coord_maps_count, num);
}

template <typename Dtype>
void ResizeWithMapLayer<Dtype>::Forward_cpu(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
	for (int i = 0; i < top.size(); ++i) {
		AffineWarpBlob_cpu(bottom[i + 2], top[i], &weights_, &locs_);
	}
}

template <typename Dtype>
void ResizeWithMapLayer<Dtype>::Backward_cpu(
		const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) 
{
	for (int i = 0; i < top.size(); ++i) {
		if (propagate_down[i + 2]) {

		  Dtype* bottom_diff = bottom[i + 2]->mutable_cpu_diff();
			const int bottom_step = bottom[i + 2]->height() * bottom[i + 2]->width();
			caffe::caffe_set(bottom[i + 2]->count(), Dtype(0.), bottom[i + 2]->mutable_cpu_diff());

		  const Dtype* top_diff = top[i]->cpu_diff();
			const int top_step = top[i]->height() * top[i]->width();

			const int count = top[i]->count();
			const int top_channels = top[i]->channels();

			const Dtype* weights_data = weights_.cpu_data();
			const int* locs_data = locs_.cpu_data();
			const int weight_num = weights_.num();
			const int weight_channels = weights_.channels();
			const int weight_height = weights_.height();
			const int weight_width = weights_.width();

			for (int index = 0; index < count; ++index) {
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
					bottom_diff[in_offset + locs_data[weights_offset + k]] +=
							(weights_data[weights_offset + k] * top_diff[index]);
				}
			}
		}
	}
}

#ifdef CPU_ONLY
STUB_GPU(ResizeWithMapLayer);
#endif

INSTANTIATE_CLASS(ResizeWithMapLayer);
REGISTER_LAYER_CLASS(ResizeWithMap);

}  // namespace caffe
