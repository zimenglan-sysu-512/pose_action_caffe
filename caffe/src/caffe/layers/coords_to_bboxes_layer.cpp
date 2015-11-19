#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/pose_estimation_layers.hpp"

namespace caffe {

template <typename Dtype>
void CoordsToBboxesLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) 
{
  this->num_ = bottom[0]->num();
  this->channels_ = bottom[0]->channels();
  this->height_ = bottom[0]->height();
  this->width_ = bottom[0]->width();

  CHECK_EQ(this->channels_, bottom[0]->count() / this->num_);
}

template <typename Dtype>
void CoordsToBboxesLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) 
{
  this->num_ = bottom[0]->num();
  this->channels_ = bottom[0]->channels();
  this->height_ = bottom[0]->height();
  this->width_ = bottom[0]->width();
  CHECK_EQ(this->channels_, bottom[0]->count() / this->num_);
  CHECK_EQ(this->channels_ % 2, 0);

  // reshape -- (x1, y1, x2, y2)
  this->new_channels_ = 4;
  top[0]->Reshape(
      this->num_,
      this->new_channels_,
      this->height_,
      this->width_
  );
}

template <typename Dtype>
void CoordsToBboxesLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) 
{
  // offset
  int pc_offset = 0;
  int bc_offset = 0;
  // bounding box
  Dtype x1, y1, x2, y2;

  const Dtype* part_coords = bottom[0]->cpu_data();
  Dtype* bbox_coords = top[0]->mutable_cpu_data();
  for(int n = 0; n < this->num_; n++) {
    // get offset
    pc_offset = bottom[0]->offset(n);
    bc_offset = top[0]->offset(n);
    // init
    x1 = part_coords[pc_offset];
    x2 = part_coords[pc_offset];
    y1 = part_coords[pc_offset + 1];
    y2 = part_coords[pc_offset + 1];

    for(int c = 0; c < this->channels_; c += 2) {
      x1 = std::min(x1, part_coords[pc_offset + c + 0]);
      y1 = std::min(y1, part_coords[pc_offset + c + 1]);

      x2 = std::max(x2, part_coords[pc_offset + c + 0]);
      y2 = std::max(y2, part_coords[pc_offset + c + 1]);
    }
    // set
    bbox_coords[bc_offset + 0] = x1;
    bbox_coords[bc_offset + 1] = y1;
    bbox_coords[bc_offset + 2] = x2;
    bbox_coords[bc_offset + 3] = y2;
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