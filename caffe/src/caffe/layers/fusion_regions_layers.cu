#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/fast_rcnn_action_layers.hpp"

namespace caffe {

template <typename Dtype>
__global__ void FusionRegion_Forward(
    const int ims_per_batch, 
    const Dtype* primary_regions_data,
    Dtype* top_fusion_data,
    const int n_classes,
    const int n_region)
{
  CUDA_KERNEL_LOOP(ipb, ims_per_batch) {
    // initial offset
    int p_offset = ipb * n_region;
    int f_offset = ipb * n_classes * n_region ;
    // union regions
    for(int cls = 0; cls < n_classes; cls++) {
      // img_ind
      // CHECK_EQ(primary_regions_data[p_offset], top_fusion_data[f_offset]);
      top_fusion_data[f_offset] = primary_regions_data[p_offset];
      f_offset++;
      // x1: min
      if(primary_regions_data[p_offset + 1] < top_fusion_data[f_offset]) {
        top_fusion_data[f_offset] = primary_regions_data[p_offset + 1];
      }
      f_offset++;
      // y1: min
      if(primary_regions_data[p_offset + 2] < top_fusion_data[f_offset]) {
        top_fusion_data[f_offset] = primary_regions_data[p_offset + 2];
      }
      f_offset++;
      // x2: max
      if(primary_regions_data[p_offset + 3] > top_fusion_data[f_offset]) {
        top_fusion_data[f_offset] = primary_regions_data[p_offset + 3];
      }
      f_offset++;
      // y2: max
      if(primary_regions_data[p_offset + 4] > top_fusion_data[f_offset]) {
        top_fusion_data[f_offset] = primary_regions_data[p_offset + 4];
      }
      f_offset++;
    }
  }
}

// max_selected_secondary_regions_inds (ims_per_batch, n_classes, 1 , 1)
// primary regions (ims_per_batch, 5, 1, 1)
// secondary regions (n_secondary_regions * ims_per_batch, 5, 1, 1)
// labels (ims_per_batch, 1, 1, 1)
template <typename Dtype>
void FusionRegionsLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) 
{
  const Dtype* cls_sr_inds_data = bottom[0]->cpu_data();
  const Dtype* primary_regions_data = bottom[1]->gpu_data();
  const Dtype* secondary_regions_data = bottom[2]->gpu_data();
  const Dtype* labels = bottom[3]->cpu_data();

  Dtype* top_fusion_data = this->max_selected_secondary_regions_.mutable_gpu_data();
  Dtype* top_fusion_data2 = NULL; 
  if(top.size() > 1) {
    top_fusion_data2 = top[1]->mutable_gpu_data();
  }
  Dtype* top_cls_data = NULL;
  if(top.size() > 2) {
    top_cls_data = top[2]->mutable_gpu_data();
  }

  const int count = bottom[0]->count();
  const int sub_count = count / this->ims_per_batch_;
  CHECK_EQ(sub_count, this->n_classes_);
  
  // secondary regions corresponding to classes, by max operation
  const int n_region = bottom[1]->count() / bottom[1]->num();
  CHECK_EQ(n_region, 5);

  for(int ipb = 0; ipb < this->ims_per_batch_; ipb++) {
    for(int cls = 0; cls < this->n_classes_; cls++) {
      const int sr_ind = cls_sr_inds_data[bottom[0]->offset(ipb, cls)];
      const int sr_offset = bottom[2]->offset(sr_ind);
      const int tf_offset = this->max_selected_secondary_regions_.offset(ipb * this->n_classes_ + cls);
      // (img_ind, x1, y1, x2, y2)
      caffe_copy(
          n_region, 
          secondary_regions_data + sr_offset, 
          top_fusion_data + tf_offset
      );
    }
  }
  // copy
  if(top.size() > 2) {
    caffe_copy(top[2]->count(), top_fusion_data, top_cls_data);
  }

  // fusion
  // NOLINT_NEXT_LINE(whitespace/operators)
  FusionRegion_Forward<Dtype><<<CAFFE_GET_BLOCKS(this->ims_per_batch_), 
    CAFFE_CUDA_NUM_THREADS>>>(
      this->ims_per_batch_, 
      primary_regions_data,
      top_fusion_data,
      this->n_classes_,
      n_region
    );

  // copy
  if(top.size() > 1) {
    caffe_copy(top[1]->count(), top_fusion_data, top_fusion_data2);
  }

  // copy
  for(int ipb = 0; ipb < this->ims_per_batch_; ipb++) {
    const int offset = bottom[3]->offset(ipb);
    const int label = labels[offset];
    const int fusion_offset = 
        this->max_selected_secondary_regions_.offset(ipb * this->n_classes_ + label);
    caffe_copy(
        n_region, 
        this->max_selected_secondary_regions_.gpu_data() + fusion_offset,
        top[0]->mutable_gpu_data() + top[0]->offset(ipb)
    );
  }
}

template <typename Dtype>
void FusionRegionsLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
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

INSTANTIATE_LAYER_GPU_FUNCS(FusionRegionsLayer);

}  // namespace caffe