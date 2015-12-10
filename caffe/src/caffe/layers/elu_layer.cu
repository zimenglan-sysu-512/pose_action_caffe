#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/person_torso_layers.hpp"

namespace caffe {

template <typename Dtype>
__global__ void ELUForward(const int n, const Dtype* in, Dtype* out,
    Dtype alpha, Dtype beta, const Dtype Zero) {
  CUDA_KERNEL_LOOP(index, n) {
    if(in[index] > Zero) {
      out[index] = in[index];
    } else {
      Dtype elu_val = alpha * (exp(in[index]) - beta);
      out[index] = elu_val <= Zero ? elu_val : Zero;
    }
  }
}

template <typename Dtype>
void ELULayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) 
{
  const int count = bottom[0]->count();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  
  const Dtype Zero = Dtype(0);
  Dtype beta = Dtype(1);
  // Dtype beta = this->layer_param_.elu_param().beta();
  Dtype alpha = this->layer_param_.elu_param().alpha();
  CHECK_GE(beta, Zero);
  CHECK_GT(alpha, Zero);

  // NOLINT_NEXT_LINE(whitespace/operators)
  ELUForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, top_data, alpha, beta, Zero);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void ELUBackward(const int n, const Dtype* in_diff,
    const Dtype* in_data, const Dtype* out_data, 
    Dtype* out_diff, Dtype alpha, Dtype beta) 
{
  CUDA_KERNEL_LOOP(index, n) {
    Dtype diff_val = (in_data[index] > 0) + 
        (in_data[index] <= 0) * (out_data[index] + alpha * beta);
    out_diff[index] = in_diff[index] * diff_val;
  }
}

template <typename Dtype>
void ELULayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) 
{
  if (propagate_down[0]) {
    const int count = bottom[0]->count();
    const Dtype* top_diff = top[0]->gpu_diff();
    const Dtype* top_data = top[0]->gpu_data();
    const Dtype* bottom_data = bottom[0]->gpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    
    Dtype beta = Dtype(1);
    // Dtype beta = this->layer_param_.elu_param().beta();
    Dtype alpha = this->layer_param_.elu_param().alpha();

    // NOLINT_NEXT_LINE(whitespace/operators)
    ELUBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, bottom_data, top_data, bottom_diff, alpha, beta);
    CUDA_POST_KERNEL_CHECK;
  }
}


INSTANTIATE_LAYER_GPU_FUNCS(ELULayer);


}  // namespace caffe
