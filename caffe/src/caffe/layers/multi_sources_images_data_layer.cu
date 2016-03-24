#include <vector>

#include "caffe/pose_estimation_layers.hpp"
#include "caffe/global_variables.hpp"

namespace caffe {

template <typename Dtype>
void MultiSourcesImagesDataLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
  // First, join the thread
  DLOG(INFO) << "\nThread joined\n";
  this->JoinPrefetchThread();

  // Used for Accuracy and Visualization layers
  // place in `Forward_cpu` to keep synchronization with top blobs
  // when use `imgidxs` and `images_paths` in other layers
  GlobalVars::set_objidxs(this->objidxs_);
  GlobalVars::set_imgidxs(this->imgidxs_);
  GlobalVars::set_images_paths(this->images_paths_);

  // Reshape to loaded data.
  top[0]->Reshape(this->prefetch_data_.num(), 
                  this->prefetch_data_.channels(),
                  this->prefetch_data_.height(), 
                  this->prefetch_data_.width());
  if(this->layer_param_.is_disp_info()) {
    LOG(INFO);
    LOG(INFO) << "top blob shape (data): " 
              << top[0]->shape_string();
    LOG(INFO);
  }
  
  // Copy the data
  caffe_copy(this->prefetch_data_.count(), 
             this->prefetch_data_.cpu_data(),
             top[0]->mutable_gpu_data());
  
  DLOG(INFO) << "\nPrefetch copied\n";
  if (this->output_labels_) {
    caffe_copy(this->prefetch_label_.count(), 
               this->prefetch_label_.cpu_data(),
               top[1]->mutable_gpu_data());
    caffe_copy(this->aux_info_.count(), 
               this->aux_info_.cpu_data(),
               top[2]->mutable_gpu_data());
    if(this->layer_param_.is_disp_info()) {
      LOG(INFO);
      LOG(INFO) << "top blob shape (labels): " 
                << top[1]->shape_string();
      LOG(INFO) << "top blob shape (labels info): " 
                << top[2]->shape_string();
      LOG(INFO);
    }
  }
  
  // Start a new prefetch thread
  DLOG(INFO) << "\nCreatePrefetchThread\n";
  this->CreatePrefetchThread();
}

INSTANTIATE_LAYER_GPU_FORWARD(MultiSourcesImagesDataLayer);

}  // namespace caffe
