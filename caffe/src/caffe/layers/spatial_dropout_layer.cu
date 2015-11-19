#include <algorithm>
#include <limits>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/pose_estimation_layers.hpp"

namespace caffe {

template <typename Dtype>
void SpatialDropoutLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();

  if(this->phase_ == TRAIN) {
    const int num = bottom[0]->num();
    const int channels = bottom[0]->channels();
    const int count = bottom[0]->count();
    const int sub_count = count / (num * channels);
    const int mask_count = rand_vec_.count();
    CHECK_EQ(mask_count, num * channels);
    CHECK_EQ(count, mask_count * sub_count);
    CHECK_EQ(bottom[0]->count(), top[0]->count());
    // Produce the random numbers
    unsigned int* mask = static_cast<unsigned int*>(rand_vec_.mutable_gpu_data());
    caffe_gpu_rng_uniform(mask_count, mask);
    caffe_copy(mask_count, rand_vec_.gpu_data(), rand_vec_.mutable_cpu_data());
    // Get pointer
    const unsigned int* mask2 = static_cast<const unsigned int*>(rand_vec_.cpu_data());
    // Start
    for(int n = 0; n < num; n++) {
      for(int c = 0; c < channels; c++) {
        const int d_offset = bottom[0]->offset(n, c);
        const int m_offset = rand_vec_.offset(n, c);
        // Dropout
        const Dtype alpha = Dtype((mask2[m_offset] > uint_thres_) * scale_);
        caffe_gpu_scale(sub_count, alpha, bottom_data + d_offset, top_data + d_offset);
      }
    }
  } else if(this->phase_ == TEST) {
    caffe_copy(bottom[0]->count(), bottom_data, top_data);
  } else {
    NOT_IMPLEMENTED;
  }
}

template <typename Dtype>
void SpatialDropoutLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();

    if(this->phase_ == TRAIN) {
      const int num = top[0]->num();
      const int channels = top[0]->channels();
      const int count = top[0]->count();
      const int sub_count = count / (num * channels);
      const int mask_count = rand_vec_.count();
      CHECK_EQ(mask_count, num * channels);
      CHECK_EQ(count, mask_count * sub_count);
      CHECK_EQ(bottom[0]->count(), top[0]->count());
      // Get pointer (note that in the `Forward_gpu`, we has use `caffe_copy` to convert data from gpu to cpu)
      const unsigned int* mask = static_cast<const unsigned int*>(rand_vec_.cpu_data());
      for(int n = 0; n < num; n++) {
        for(int c = 0; c < channels; c++) {
          const int d_offset = top[0]->offset(n, c);
          const int m_offset = rand_vec_.offset(n, c);
          // Dropout
          const Dtype alpha = Dtype((mask[m_offset] > uint_thres_) * scale_);
          caffe_gpu_scale(sub_count, alpha, top_diff + d_offset, bottom_diff + d_offset);
        }
      }
    } else if(this->phase_ == TEST) {
      caffe_copy(top[0]->count(), top_diff, bottom_diff);
    } else {
      NOT_IMPLEMENTED;
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SpatialDropoutLayer);


}  // namespace caffe
