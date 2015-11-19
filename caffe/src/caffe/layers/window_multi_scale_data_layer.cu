// CopyRight ZhuJin Liang 2015

#include <vector>

#include "caffe/global_variables.hpp"
#include "caffe/wanglan_face_shoulders_layers.hpp"

namespace caffe {

template <typename Dtype>
void WindowMultiScaleDataLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
  /// Prepare
  if(cur_batch_use_count < batch_reuse_count) {
    ++cur_batch_use_count;
    return;
  }
  cur_batch_use_count = 1;

  /// First, join the thread
  this->JoinPrefetchThread();

  /// Used for Accuracy and Visualization layers
  /// place in `Forward_cpu` to keep synchronization with top blobs
  /// when use `imgidxs` and `images_paths` in other layers
  // GlobalVars::set_objidxs(this->objidxs_);
  // GlobalVars::set_imgidxs(this->imgidxs_);
  // GlobalVars::set_images_paths(this->images_paths_);

  /// Reshape to loaded data.
  top[0]->Reshape(
      this->prefetch_data_->num(), 
      this->prefetch_data_->channels(),
      this->prefetch_data_->height(), 
      this->prefetch_data_->width()
  );
  /// Copy the data
  caffe_copy(
      this->prefetch_data_->count(), 
      this->prefetch_data_->cpu_data(),
      top[0]->mutable_gpu_data()
  );
  /// Copy the labels
  caffe_copy(
      this->prefetch_label_->count(), 
      this->prefetch_label_->cpu_data(),
      top[1]->mutable_gpu_data()
  );
  /// Copy the bboxes
  if (this->with_bbox_) {
    caffe_copy(
        this->prefetch_bbox_->count(), 
        this->prefetch_bbox_->cpu_data(),
        top[2]->mutable_gpu_data()
    );
  }

  /// Start a new prefetch thread
  this->CreatePrefetchThread();
}

INSTANTIATE_LAYER_GPU_FORWARD(WindowMultiScaleDataLayer);

}  // namespace caffe
