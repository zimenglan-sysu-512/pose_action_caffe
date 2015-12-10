#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/person_torso_layers.hpp"

namespace caffe {

template <typename Dtype>
inline Dtype elu_func(const Dtype x, const Dtype alpha, 
    const Dtype beta = Dtype(1)) 
{
  return alpha * (exp(x) - beta);
}

template <typename Dtype>
void ELULayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) 
{
  const int count = bottom[0]->count();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const Dtype* bottom_data = bottom[0]->cpu_data();

  const Dtype Zero = Dtype(0);
  Dtype beta = Dtype(1);
  // Dtype beta = this->layer_param_.elu_param().beta();
  Dtype alpha = this->layer_param_.elu_param().alpha();
  CHECK_GE(beta, Zero);
  CHECK_GT(alpha, Zero);

  for (int i = 0; i < count; ++i) {
    top_data[i] = bottom_data[i] > Zero ?
        bottom_data[i] : 
        std::min(elu_func(bottom_data[i], alpha, beta), Zero);
  }
}

template <typename Dtype>
void ELULayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) 
{
  if (propagate_down[0]) {
    const int count = bottom[0]->count();
    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* top_data = top[0]->cpu_data();
    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

    Dtype beta = Dtype(1);
    // Dtype beta = this->layer_param_.elu_param().beta();
    Dtype alpha = this->layer_param_.elu_param().alpha();

    for (int i = 0; i < count; ++i) {
      Dtype diff_val = (bottom_data[i] > 0) + 
          (bottom_data[i] <= 0) * (top_data[i] + alpha * beta);
      bottom_diff[i] = top_diff[i] * diff_val;
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(ELULayer);
#endif

INSTANTIATE_CLASS(ELULayer);

}  // namespace caffe
