#include <vector>

#include "caffe/layers/zhouyan_ann_layers.hpp"
#include "caffe/global_variables.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void LoadDataFromFileLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
  // First, join the thread
  DLOG(INFO) << "\nThread joined\n";
  this->JoinPrefetchThread();

  top[0]->Reshape(this->prefetch_data_.num(), 
                  this->prefetch_data_.channels(),
                  this->prefetch_data_.height(), 
                  this->prefetch_data_.width());
  caffe_scal(this->prefetch_data_.count(), this->scale_, 
             this->prefetch_data_.mutable_cpu_data());
  caffe_copy(this->prefetch_data_.count(), 
             this->prefetch_data_.cpu_data(),
             top[0]->mutable_gpu_data());
  if(this->layer_param_.is_disp_info()) {
    LOG(INFO) << "Copy and Reshape top 0 blob (data)...";
  }
  
  DLOG(INFO) << "\nPrefetch copied\n";
  if (this->output_labels_) {
    top[1]->Reshape(this->prefetch_label_.num(), 
                    this->prefetch_label_.channels(),
                    this->prefetch_label_.height(), 
                    this->prefetch_label_.width());
    caffe_scal(this->prefetch_label_.count(), this->scale_, 
               this->prefetch_label_.mutable_cpu_data());
    caffe_copy(this->prefetch_label_.count(), 
               this->prefetch_label_.cpu_data(),
               top[1]->mutable_gpu_data());
    if(this->layer_param_.is_disp_info()) {
      LOG(INFO) << "Copy and Reshape top 1 blob (labels)...";
    }
  }
  
  // Start a new prefetch thread
  DLOG(INFO) << "\nCreatePrefetchThread\n";
  this->CreatePrefetchThread();
}

INSTANTIATE_LAYER_GPU_FORWARD(LoadDataFromFileLayer);

}  // namespace caffe
