#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/pose_estimation_layers.hpp"

namespace caffe {

template <typename Dtype>
void CoordsToBboxesLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
  const CoordsToBboxesParameter coord2bbox_param = 
      this->layer_param_.coord2bbox_param();
  CHECK(coord2bbox_param.has_bbox_id1());
  CHECK(coord2bbox_param.has_bbox_id2());
  this->bbox_id1_ = coord2bbox_param.bbox_id1();
  this->bbox_id2_ = coord2bbox_param.bbox_id2();

  this->as_whole_ = false;
  if((this->bbox_id1_ == this->bbox_id2_) || 
      (this->bbox_id1_ < 0 && this->bbox_id2_ < 0)) {
    this->as_whole_ = true;
  }
  if(this->bbox_id1_ > this->bbox_id2_) {
    std::swap(this->bbox_id1_, this->bbox_id2_);
  }
}

template <typename Dtype>
void CoordsToBboxesLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
  const int num      = bottom[0]->num();
  const int channels = bottom[0]->channels();
  const int count    = bottom[0]->count();
  // const int height   = bottom[0]->height();
  // const int width    = bottom[0]->width();

  CHECK_EQ(channels % 2, 0);
  CHECK_EQ(channels, count / num);

  top[0]->Reshape(num, 4, 1, 1);
}

template <typename Dtype>
void CoordsToBboxesLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
  int pc_offset = 0;
  int bc_offset = 0;
  Dtype x1, y1, x2, y2;

  const Dtype* part_coords = bottom[0]->cpu_data();
  Dtype* bbox_coords       = top[0]->mutable_cpu_data();

  if(this->as_whole_) {
    for(int n = 0; n < bottom[0]->num(); n++) {
      pc_offset = bottom[0]->offset(n);
      bc_offset = top[0]->offset(n);

      x1 = part_coords[pc_offset];
      x2 = part_coords[pc_offset];
      y1 = part_coords[pc_offset + 1];
      y2 = part_coords[pc_offset + 1];

      for(int c = 0; c < bottom[0]->channels(); c += 2) {
        x1 = std::min(x1, part_coords[pc_offset + c + 0]);
        x2 = std::max(x2, part_coords[pc_offset + c + 0]);

        y1 = std::min(y1, part_coords[pc_offset + c + 1]);
        y2 = std::max(y2, part_coords[pc_offset + c + 1]);
      }

      bbox_coords[bc_offset + 0] = x1;
      bbox_coords[bc_offset + 1] = y1;
      bbox_coords[bc_offset + 2] = x2;
      bbox_coords[bc_offset + 3] = y2;
    }
  } else {
    int idx1 = this->bbox_id1_ * 2;
    int idx2 = this->bbox_id2_ * 2;
    for(int n = 0; n < bottom[0]->num() ; n++) {
      pc_offset = bottom[0]->offset(n);
      bc_offset = top[0]->offset(n);

      x1 = part_coords[pc_offset + idx1 + 0];
      y1 = part_coords[pc_offset + idx1 + 1];
      
      x2 = part_coords[pc_offset + idx2 + 0];
      y2 = part_coords[pc_offset + idx2 + 1];

      if(x1 > x2) std::swap(x1, x2);
      if(y1 > y2) std::swap(y1, y2);

      bbox_coords[bc_offset + 0] = x1;
      bbox_coords[bc_offset + 1] = y1;
      bbox_coords[bc_offset + 2] = x2;
      bbox_coords[bc_offset + 3] = y2;
    }
  }
}

template <typename Dtype>
void CoordsToBboxesLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
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
STUB_GPU(CoordsToBboxesLayer);
#endif

INSTANTIATE_CLASS(CoordsToBboxesLayer);
REGISTER_LAYER_CLASS(CoordsToBboxes);

}  // namespace caffe