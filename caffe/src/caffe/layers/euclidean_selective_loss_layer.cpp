// Copyright 2015 Zhu.Jin Liang


#include <vector>
#include <utility>
#include <algorithm>

#include "caffe/layer.hpp"
#include "caffe/common.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/util/math_functions.hpp"

#include "caffe/zhu_face_layers.hpp"
#include "caffe/util/util_pre_define.hpp"

namespace caffe {

template <typename Dtype>
void EuclideanSelectiveLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
      << "Inputs must have the same dimension.";
  if (bottom.size() > 2) {
    CHECK_EQ(bottom[1]->count(1), bottom[2]->count(1))
        << "Mask and input must have the same dimension.";
  }
  diff_.ReshapeLike(*(bottom[0]));
  mask_.ReshapeLike(*(bottom[0]));
  caffe::caffe_set(mask_.count(), Dtype(1), mask_.mutable_cpu_data());

  CHECK(this->layer_param_.has_euclidean_selective_loss_param());
  CHECK(this->layer_param_.euclidean_selective_loss_param().has_force_binary());
  force_binary_ = this->layer_param_.euclidean_selective_loss_param().force_binary();
  if (force_binary_) {
  	binary_threshold_ = this->layer_param_.euclidean_selective_loss_param().binary_threshold();
    threshold_top_.ReshapeLike(*(bottom[1]));
  }
}

template <typename Dtype>
void EuclideanSelectiveLossLayer<Dtype>::Forward_cpu(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
	const EuclideanSelectiveLossParameter euclidean_selective_loss_param = 
			this->layer_param_.euclidean_selective_loss_param();
  // 负样本最多传正样本的neg_factor倍
	const float neg_factor = euclidean_selective_loss_param.neg_factor();
  // 负样本里面，hard sample的比例
  const float hard_factor = MIN(1.0f, euclidean_selective_loss_param.hard_factor());
	const float hard_threshold = euclidean_selective_loss_param.hard_threshold();
  // 假设没有正样本在，那么负样本是占所有样本的比例
  const float neg_factor_without_pos = MIN(1.0f, euclidean_selective_loss_param.neg_factor_without_pos());
  const float mask_threshold = MIN(1.0f, euclidean_selective_loss_param.mask_threshold());

	const Dtype* input = bottom[0]->cpu_data();
	const Dtype* target = bottom[1]->cpu_data();
	Dtype* diff = diff_.mutable_cpu_data();
	caffe::caffe_set(diff_.count(), Dtype(0), diff);
	const Dtype* mask = bottom.size() > 2 ? bottom[2]->cpu_data() : mask_.cpu_data();

	if (force_binary_) {
		caffe_cpu_threshold(
				bottom[1]->count(), 
				binary_threshold_,
				bottom[1]->cpu_data(),
				threshold_top_.mutable_cpu_data()
		);
		// Reset
		target = threshold_top_.cpu_data();
	}

	const int num = bottom[0]->num();
	const int channels = bottom[0]->channels();
	// 按height * width来
	const int dims = bottom[0]->height() * bottom[0]->width();

	Dtype loss = 0;
	for (int n = 0; n < num; ++n) {
		// 在训练侧脸的时候, 比如左侧脸, 这时候右眼就不会存在
		// 其实是不应该对这个channels做惩罚, 调整一下策略
		std::vector<std::vector<std::pair<Dtype, int> > > neg_idx_all;
		std::vector<int> hard_count_all;
	  bool hasPos = false;

	  for (int c = 0; c < channels; ++c) {
			const int offset = (n * channels + c) * dims;
			for (int i = 0; i < dims; ++i) {
				const int idx = offset + i;
				Dtype dot = input[idx] - target[idx];
				dot = dot * dot;
				loss = loss + dot;
			}

			if (this->net_->phase() == caffe::TEST) continue;

			int pos_count = 0;
			int hard_count = 0;
			neg_idx.clear();
			for (int i = 0; i < dims; ++i) {
				const int idx = offset + i;
				if (!(target[idx] > 0)) {
					if (mask == NULL || !(mask[idx] < mask_threshold)) {
						neg_idx.push_back(make_pair(input[idx], idx));

						if (input[idx] >= hard_threshold) {
							++hard_count;
						}
					}
				} else {
					++pos_count;
					diff[idx] = (input[idx] - target[idx]) * mask[idx];
				}
			}

			if (pos_count == 0 && !hasPos) {
				neg_idx_all.push_back(neg_idx);
				hard_count_all.push_back(hard_count);
				continue;
			}
			hasPos = true;
			int pass_neg_count = floor(pos_count * neg_factor);
			pass_neg_count = MIN(pass_neg_count, neg_idx.size());

			// 先对neg_idx按照score排序
			std::stable_sort(neg_idx.begin(), neg_idx.end(), std::greater<pair<Dtype, int> >());
			// 先打乱所有的hard sample，
			// 然后前pass_hard_count为最终传梯度的hard sample
			shuffle(neg_idx.begin(), neg_idx.begin() + hard_count);

			// 接着除去pass_hard_count这么多个hard sample，剩下的作为普通的负样本
			// 对这些普通负样本进行打乱，然后按顺序选前面的出来传梯度
			const int pass_hard_count = pass_neg_count * hard_factor;
			shuffle(neg_idx.begin() + pass_hard_count, neg_idx.end());

			for (int i = 0; i < pass_neg_count; ++i) {
				const int idx = neg_idx[i].second;
				diff[idx] = (input[idx] - target[idx]) * mask[idx];
			}
		}

		if (!hasPos) {
			for (int n = 0; n < neg_idx_all.size(); ++n) {
				if (neg_idx_all[n].size() == 0) continue;

				const int pass_neg_count = MIN(neg_idx_all[n].size(),
						neg_idx_all[n].size() * neg_factor_without_pos);

				// 先对neg_idx按照score排序
				std::stable_sort(neg_idx_all[n].begin(), neg_idx_all[n].end(), std::greater<pair<Dtype, int> >());
				// 先打乱所有的hard sample，
				// 然后前pass_hard_count为最终传梯度的hard sample
				shuffle(neg_idx_all[n].begin(), neg_idx_all[n].begin() + hard_count_all[n]);
				const int pass_hard_count = pass_neg_count * hard_factor;

				// 接着除去pass_hard_count这么多个hard sample，剩下的作为普通的负样本
				// 对这些普通负样本进行打乱，然后按顺序选前面的出来传梯度
				shuffle(neg_idx_all[n].begin() + pass_hard_count, neg_idx_all[n].end());

				for (int i = 0; i < pass_neg_count; ++i) {
					const int idx = neg_idx_all[n][i].second;
					diff[idx] = (input[idx] - target[idx]) * mask[idx];
				}
			}
		}
	}

  loss = loss / bottom[0]->num() / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void EuclideanSelectiveLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
      caffe_cpu_axpby(
          bottom[i]->count(),                 // count
          alpha,                              // alpha
          diff_.cpu_data(),                   // a
          Dtype(0),                           // beta
          bottom[i]->mutable_cpu_diff());     // b
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(EuclideanSelectiveLossLayer);
#endif

INSTANTIATE_CLASS(EuclideanSelectiveLossLayer);
REGISTER_LAYER_CLASS(EuclideanSelectiveLoss);

}  // namespace caffe