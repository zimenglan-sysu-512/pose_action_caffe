#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/fast_rcnn_action_layers.hpp"

namespace caffe {

template <typename Dtype>
void CrossBlobsSumLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
  const int num = bottom[0]->num();
  const int channels = bottom[0]->channels();
  const int height = bottom[0]->height();
  const int width = bottom[0]->width();
  const int size = bottom.size();

  for(int idx = 0; idx < size; idx++) {
    CHECK_EQ(num, bottom[idx]->num());
    CHECK_EQ(channels, bottom[idx]->channels());
    CHECK_EQ(height, bottom[idx]->height());
    CHECK_EQ(width, bottom[idx]->width());
  }
}

template <typename Dtype>
void CrossBlobsSumLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
  const int num = bottom[0]->num();
  const int channels = bottom[0]->channels();
  const int height = bottom[0]->height();
  const int width = bottom[0]->width();

  top[0]->Reshape(num, channels, height, width);
  for(int idx = 0; idx < bottom.size(); idx++) {
    CHECK_EQ(bottom[idx]->count(), top[0]->count());
  }
}

template <typename Dtype>
void CrossBlobsSumLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
  // 
  const int count = top[0]->count();
  Dtype* top_data = top[0]->mutable_cpu_data();

  // set to be zero
  const Dtype zero = Dtype(0);
  caffe_set(count, zero, top_data);

  // 
  const Dtype alpha = Dtype(1);
  const Dtype beta = Dtype(1);

  for(int idx = 0; idx < bottom.size(); idx++) {
    const Dtype* bottom_data = bottom[idx]->cpu_data();
    caffe_cpu_axpby( 
        count,
        alpha,
        bottom_data,
        beta,
        top_data
    );
  }
}

template <typename Dtype>
void CrossBlobsSumLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, 
    const vector<Blob<Dtype>*>& bottom) 
{
  if(!propagate_down[0]) {
    return;
  }
  const int count = top[0]->count();
  const Dtype* top_diff = top[0]->cpu_diff();

  for (int idx = 0; idx < bottom.size(); idx++) {
    // if (!propagate_down[idx]) { continue; }
    // need to be reset bottom_diff ??
    Dtype* bottom_diff = bottom[idx]->mutable_cpu_diff();
    caffe_copy(
        count, 
        top_diff,
        bottom_diff
    );
  } 
}

#ifdef CPU_ONLY
STUB_GPU(CrossBlobsSumLayer);
#endif

INSTANTIATE_CLASS(CrossBlobsSumLayer);
REGISTER_LAYER_CLASS(CrossBlobsSum);

}  // namespace caffe
