#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/fast_rcnn_action_layers.hpp"

namespace caffe {

// max_selected_secondary_regions_inds (ims_per_batch, n_classes, 1 , 1)
// primary regions (ims_per_batch, 5, 1, 1)
// secondary regions (n_secondary_regions * ims_per_batch, 5, 1, 1)
template <typename Dtype>
void FusionRegionsLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) 
{
  CHECK_EQ(bottom[0]->num(), bottom[1]->num());
  CHECK_EQ((bottom[2]->num() % bottom[1]->num()), 0);

  CHECK_EQ(bottom[1]->channels(), bottom[2]->channels());
  CHECK_EQ(bottom[1]->height(), bottom[2]->height());
  CHECK_EQ(bottom[1]->width(), bottom[2]->width());

  CHECK_EQ(bottom[1]->num(), bottom[3]->num());
  CHECK_EQ(bottom[3]->count(), bottom[3]->num());
}

// bottom[0]: max_selected_secondary_regions_inds
// bottom[1]: primary regions
// bottom[2]: secondary regions
// bottom[3]: labels
// top[0]: fused_primary_regions (only ground truth class)
// top[1]: fused_primary_regions (all classes), if have
// top[2]: all corresponding max-selected secondary regions, if have
template <typename Dtype>
void FusionRegionsLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) 
{
  CHECK_EQ(bottom[0]->num(), bottom[1]->num());
  CHECK_EQ((bottom[2]->num() % bottom[1]->num()), 0);

  CHECK_EQ(bottom[1]->channels(), bottom[2]->channels());
  CHECK_EQ(bottom[1]->height(), bottom[2]->height());
  CHECK_EQ(bottom[1]->width(), bottom[2]->width());

  CHECK_EQ(bottom[1]->num(), bottom[3]->num());
  CHECK_EQ(bottom[3]->count(), bottom[3]->num());

  this->ims_per_batch_ = bottom[0]->num();
  this->n_classes_ = bottom[0]->count() / bottom[0]->num();

  // pick the secondary regions w.r.t. primary regions
  // format: [..., (img_ind, x1, y1, x2, y2), ...]
  top[0]->Reshape(
      this->ims_per_batch_,
      bottom[1]->channels(),
      bottom[1]->height(),
      bottom[1]->width()
  );
  this->max_selected_secondary_regions_.Reshape(
      this->ims_per_batch_ * this->n_classes_,
      bottom[1]->channels(),
      bottom[1]->height(),
      bottom[1]->width()
  );
  if(top.size() > 1) {
    top[1]->Reshape(
        this->ims_per_batch_ * this->n_classes_,
        bottom[1]->channels(),
        bottom[1]->height(),
        bottom[1]->width()
    );
  }
  if(top.size() > 2) {
    top[2]->Reshape(
        this->ims_per_batch_ * this->n_classes_,
        bottom[1]->channels(),
        bottom[1]->height(),
        bottom[1]->width()
    );
  }
}

template <typename Dtype>
void FusionRegionsLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) 
{
  const Dtype* cls_sr_inds_data = bottom[0]->cpu_data();
  const Dtype* primary_regions_data = bottom[1]->cpu_data();
  const Dtype* secondary_regions_data = bottom[2]->cpu_data();
  const Dtype* labels = bottom[3]->cpu_data();

  Dtype* top_fusion_data = this->max_selected_secondary_regions_.mutable_cpu_data();
  Dtype* top_fusion_data2 = NULL; 
  if(top.size() > 1) {
    top_fusion_data2 = top[1]->mutable_cpu_data();
  }
  Dtype* top_cls_data = NULL;
  if(top.size() > 2) {
    top_cls_data = top[2]->mutable_cpu_data();
  }

  const int count = bottom[0]->count();
  const int sub_count = count / this->ims_per_batch_;
  CHECK_EQ(sub_count, this->n_classes_);
  
  // secondary regions corresponding to classes, by max operation
  const int n_region = bottom[1]->count() / bottom[1]->num();
  CHECK_EQ(n_region, 5);

  // __asm__("int $3");
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

  // __asm__("int $3");
  // fusion
  int p_offset = 0;
  int f_offset = 0;
  for(int ipb = 0; ipb < this->ims_per_batch_; ipb++) {
    for(int cls = 0; cls < this->n_classes_; cls++) {
      // img_ind
      CHECK_EQ(primary_regions_data[p_offset], top_fusion_data[f_offset]);
      top_fusion_data[f_offset] = primary_regions_data[p_offset];
      f_offset++;
      // x1: min
      top_fusion_data[f_offset] = 
          std::min(primary_regions_data[p_offset + 1], top_fusion_data[f_offset]);
      f_offset++;
      // y1: min
      top_fusion_data[f_offset] = 
          std::min(primary_regions_data[p_offset + 2], top_fusion_data[f_offset]);
      f_offset++;
      // x2: max
      top_fusion_data[f_offset] = 
          std::max(primary_regions_data[p_offset + 3], top_fusion_data[f_offset]);
      f_offset++;
      // y2: max
      top_fusion_data[f_offset] = 
          std::max(primary_regions_data[p_offset + 4], top_fusion_data[f_offset]);
      f_offset++;
    }

    p_offset += n_region;
  }
  CHECK_EQ(p_offset, bottom[1]->count());
  CHECK_EQ(f_offset, this->max_selected_secondary_regions_.count());

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
        this->max_selected_secondary_regions_.cpu_data() + fusion_offset,
        top[0]->mutable_cpu_data() + top[0]->offset(ipb)
    );
  }
}

template <typename Dtype>
void FusionRegionsLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) 
{
  const Dtype Zero = Dtype(0);
  CHECK_EQ(propagate_down.size(), bottom.size());

  for (int i = 0; i < propagate_down.size(); ++i) {
    if (propagate_down[i]) { 
      // NOT_IMPLEMENTED; 
      caffe_set(bottom[i]->count(), Zero, bottom[i]->mutable_cpu_diff());
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(FusionRegionsLayer);
#endif

INSTANTIATE_CLASS(FusionRegionsLayer);
REGISTER_LAYER_CLASS(FusionRegions);

}  // namespace caffe