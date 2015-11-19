#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/fast_rcnn_action_layers.hpp"


namespace caffe {



template <typename Dtype>
void CrossBlobsSumLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) 
{
  // 
  const int count = top[0]->count();
  Dtype* top_data = top[0]->mutable_gpu_data();

  // set to be zero
  const Dtype zero = Dtype(0);
  caffe_gpu_set(count, zero, top_data);

  // 
  const Dtype alpha = Dtype(1);
  const Dtype beta = Dtype(1);

  for(int idx = 0; idx < bottom.size(); idx++) {
    const Dtype* bottom_data = bottom[idx]->gpu_data();
    caffe_gpu_axpby( 
        count,
        alpha,
        bottom_data,
        beta,
        top_data
    );
  }
}



template <typename Dtype>
void CrossBlobsSumLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) 
{
  // 
  if(!propagate_down[0]) {
    return;
  }
  const int count = top[0]->count();
  const Dtype* top_diff = top[0]->gpu_diff();

  for (int idx = 0; idx < bottom.size(); idx++) {
    // 
    // if (!propagate_down[idx]) { continue; }

    // need to be reset bottom_diff ??
    Dtype* bottom_diff = bottom[idx]->mutable_gpu_diff();
    caffe_copy(
        count, 
        top_diff,
        bottom_diff
    );
  } 
}


INSTANTIATE_LAYER_GPU_FUNCS(CrossBlobsSumLayer);


}  // namespace caffe
