// Copyright 2015 DDK

#include "caffe/pose_estimation_layers.hpp"

namespace caffe {

template <typename Dtype>
void TorsoMaskFromCoordsLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) 
{
  this->Forward_cpu(bottom, top);
}

template <typename Dtype>
void TorsoMaskFromCoordsLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) 
{
  const Dtype Zero = Dtype(0);
  CHECK_EQ(propagate_down.size(), bottom.size());

  for (int i = 0; i < propagate_down.size(); ++i) {
    if (propagate_down[i]) { 
      // NOT_IMPLEMENTED; 
      caffe_gpu_set(bottom[i]->count(), Zero, bottom[i]->mutable_gpu_diff());
    }
  }
}

// INSTANTIATE_LAYER_GPU_FORWARD
INSTANTIATE_LAYER_GPU_FUNCS(TorsoMaskFromCoordsLayer);

}  // namespace caffe
