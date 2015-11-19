// Copyright 2015 DDK (dongdk.sysu@foxmail.com)

#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/pose_estimation_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/common.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

using std::min;
using std::max;

template <typename Dtype>
void HeatMapsFromCoordsLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
  CHECK(this->layer_param_.has_heat_maps_from_coords_param());
  const HeatMapsFromCoordsParameter heat_maps_from_coords_param = 
      this->layer_param_.heat_maps_from_coords_param();

  CHECK(heat_maps_from_coords_param.has_is_binary());
  CHECK(heat_maps_from_coords_param.has_heat_map_a());
  CHECK(heat_maps_from_coords_param.has_heat_map_b());
  CHECK(heat_maps_from_coords_param.has_valid_dist_factor());

  this->is_binary_ = heat_maps_from_coords_param.is_binary(); 
  this->heat_map_a_  = heat_maps_from_coords_param.heat_map_a();
  this->heat_map_b_  = heat_maps_from_coords_param.heat_map_b();
  this->valid_dist_factor_  = heat_maps_from_coords_param.valid_dist_factor();

  LOG(INFO) << "*************************************************";
  LOG(INFO) << "is_binary: " << this->is_binary_;
  LOG(INFO) << "heat_map_a: " << this->heat_map_a_;
  LOG(INFO) << "heat_map_b: " << this->heat_map_b_;
  LOG(INFO) << "valid_dist_factor: " << this->valid_dist_factor_;
  LOG(INFO) << "**************************************************";
}

template <typename Dtype>
void HeatMapsFromCoordsLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
  this->batch_num_ = bottom[0]->num();
  this->label_num_ = bottom[0]->count() / this->batch_num_;
  this->key_point_num_ = this->label_num_ / 2;
  CHECK_EQ(this->label_num_, this->key_point_num_ * 2);
  CHECK_EQ(this->label_num_, bottom[0]->channels());

  // aux info (img_ind, width, height, im_scale, flippable)
  CHECK_EQ(bottom[1]->channels(), 5);
  CHECK_EQ(bottom[1]->num(), bottom[0]->num());
  CHECK_EQ(bottom[1]->channels(), bottom[1]->count() / bottom[1]->num());

  const Dtype* aux_info = bottom[1]->cpu_data();
  Dtype max_width = Dtype(-1);
  Dtype max_height = Dtype(-1);
  for(int item_id = 0; item_id < this->batch_num_; item_id++) {
    // (img_ind, width, height, im_scale, flippable)
    const int offset = bottom[1]->offset(item_id);
    // const Dtype img_ind   = aux_info[offset + 0];
    const Dtype width     = aux_info[offset + 1];
    const Dtype height    = aux_info[offset + 2];
    const Dtype im_scale  = aux_info[offset + 3];
    // const Dtype flippable = aux_info[offset + 4];
    max_width = std::max(max_width, width * im_scale);
    max_height = std::max(max_height, height * im_scale);
  }
  // reshape & reset
  const int mod_w = int(max_width) % this->heat_map_a_;
  if(mod_w > 0) max_width = max_width - mod_w + this->heat_map_a_;
  const int mod_h = int(max_height) % this->heat_map_a_;
  if(mod_h > 0) max_height = max_height - mod_h + this->heat_map_a_;
  
  /// I don't know whether has heat_map_b_ is good or not, 
  /// It looks like a translated factor...
  // CHECK_EQ(this->heat_map_b_, 0) << "here we don't need heat_map_b_, just set it to be 0...";
  this->max_width_ = max_width;
  this->max_height_ = max_height;
  CHECK_GT(this->heat_map_a_, 0);
  CHECK_LT(this->heat_map_a_, max_width);
  CHECK_LT(this->heat_map_a_, max_height);
  CHECK_GE(this->heat_map_b_, 0);
  CHECK_LT(this->heat_map_b_, max_width);
  CHECK_LT(this->heat_map_b_, max_height);
  this->heat_width_ = (this->max_width_ - this->heat_map_b_) / this->heat_map_a_ + 1;
  this->heat_height_ = (this->max_height_ - this->heat_map_b_) / this->heat_map_a_ + 1;
  this->heat_num_ = this->heat_width_ * this->heat_height_;

  // top blob
  top[0]->Reshape(this->batch_num_, this->key_point_num_, this->heat_height_, this->heat_width_);
  // prefetch_coordinates_labels
  this->prefetch_heat_maps_.Reshape(this->batch_num_, this->key_point_num_, this->heat_height_, this->heat_width_);
  // mask
  top[1]->Reshape(this->batch_num_, this->key_point_num_, 1, 1);
  this->prefetch_heat_maps_masks_.Reshape(this->batch_num_, this->key_point_num_, 1, 1); 

  // (heat_width, heat_height, heat_map_a, heat_map_b)
  if(top.size() == 3) {
    top[2]->Reshape(4, 1, 1, 1);
    top[2]->mutable_cpu_data()[0] = Dtype(this->heat_width_);
    top[2]->mutable_cpu_data()[1] = Dtype(this->heat_height_);
    top[2]->mutable_cpu_data()[2] = Dtype(this->heat_map_a_);
    top[2]->mutable_cpu_data()[3] = Dtype(this->heat_map_b_);
  }
}

template <typename Dtype> 
void HeatMapsFromCoordsLayer<Dtype>::CreateHeatMapsFromCoords(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
  const Dtype One = Dtype(1);
  const Dtype Zero = Dtype(0);
  const Dtype* coords_ptr = bottom[0]->cpu_data();
  // 
  Dtype* heat_maps_ptr = this->prefetch_heat_maps_.mutable_cpu_data();
  Dtype* heat_maps_masks_ptr = this->prefetch_heat_maps_masks_.mutable_cpu_data();
  // 
  caffe_set(this->prefetch_heat_maps_.count(), Zero, this->prefetch_heat_maps_.mutable_cpu_data());
  caffe_set(this->prefetch_heat_maps_masks_.count(), One, this->prefetch_heat_maps_masks_.mutable_cpu_data());

  int kpn_count;
  int coords_offset;
  int heat_map_masks_offset;
  // control the affected area
  const int valid_len = round(this->heat_map_a_ * this->valid_dist_factor_);
  const Dtype valid_square = valid_len * valid_len; 

  // start
  const Dtype* aux_info = bottom[1]->cpu_data();
  for(int bn = 0; bn < this->batch_num_; bn++) {
    // get offset of coordinates
    coords_offset = bottom[0]->offset(bn);
    /// (img_ind, width, height, im_scale, flippable)
    const int info_offset = bottom[1]->offset(bn);
    /// const Dtype img_ind   = aux_info[info_offset + 0];
    const Dtype o_width     = aux_info[info_offset + 1];
    const Dtype o_height    = aux_info[info_offset + 2];
    const Dtype im_scale  = aux_info[info_offset + 3];
    const Dtype r_width = o_width * im_scale;
    const Dtype r_height = o_height * im_scale;
    /// const Dtype flippable = aux_info[info_offset + 4];

    for(int kpn = 0; kpn < this->key_point_num_; kpn++) {
      kpn_count = 0;
      const int idx = kpn * 2;
      // origin coordinates for one part (x, y)
      const Dtype ox = coords_ptr[coords_offset + idx];
      const Dtype oy = coords_ptr[coords_offset + idx + 1];
      // invalid coordinates
      if(ox < Zero || oy < Zero || ox >= r_width|| oy >= r_height) { 
        heat_map_masks_offset = this->prefetch_heat_maps_masks_.offset(bn, kpn);
        heat_maps_masks_ptr[heat_map_masks_offset] = Zero;
        continue; 
      }

      // mapping: heat map's coordinates for one part (hx, hy)
      const Dtype hx = round((ox - this->heat_map_b_) / Dtype(this->heat_map_a_ + 0.));
      const Dtype hy = round((oy - this->heat_map_b_) / Dtype(this->heat_map_a_ + 0.));

      // coordinates around (hx, hy)
      for(int oj = -valid_len; oj <= valid_len; oj++) {
        const Dtype hy2 = hy + oj;
        if(hy2 < 0 || hy2 >= this->heat_height_) { continue; }

        for(int oi = -valid_len; oi <= valid_len; oi++) {
          const Dtype hx2 = hx + oi;
          if(hx2 < 0 || hx2 >= this->heat_width_) { continue; }
          const Dtype square = oi * oi + oj * oj;
          if(square > valid_square) { continue; }
          kpn_count++;
          // get offset for heat maps
          const int h_idx = this->prefetch_heat_maps_.offset(bn, kpn, int(hy2), int(hx2));

          if(this->is_binary_) {
            heat_maps_ptr[h_idx] = One;
          } else {
            /*
              hx1 = (ox - a) / b, hy1 = (oy - a) / b
              dx = hx2 * a + b - ox = (hx1 + oi) * a + b - ox = (hx1 * a + b - ox) + oi * a ~~ oi * a
              dy = hy2 * a + b - oy = (hy1 + oj) * a + b - oy = (hy1 * a + b - oy) + oj * a ~~ oj * a
              oi ~ [-valid_len, valid_len]
              oj ~ [-valid_len, valid_len]
              means:
                heat map会损失coordinates的精度,因为原图到heat map的转换，heat map到原图的转换,
                会导致coordinates有着0.5*scale*heat_map_a的偏差(这里将heat_map_b设为0)
              dist in [1/(e^2), 1]
            */
            // f(x, y) = (1 / 2 * pi * std_x * std_y) * exp(-1/2 * ((x - mean_x)^2 / std_x^2 + (y - mean_y)^2 / std_y^2))
            // mean_x = mean_y = 0, std_x = std_y = 1.5
            // for simplicity, f(x, y) = exp(-4.5 * (x^2 + y^2))
            // const Dtype dist = exp(- 2 * square / valid_square);
            const Dtype dist = exp(- square / valid_len);
            heat_maps_ptr[h_idx] = max(heat_maps_ptr[h_idx], dist);
          }
        }
      }

      if(kpn_count <= 0) {
        heat_map_masks_offset = this->prefetch_heat_maps_masks_.offset(bn, kpn);
        heat_maps_masks_ptr[heat_map_masks_offset] = Zero;
      }
    }
  }
}

template <typename Dtype>
void HeatMapsFromCoordsLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
  // Get Heat maps for batch
  CreateHeatMapsFromCoords(bottom, top);
  // Copy the heat maps
  caffe_copy(top[0]->count(), this->prefetch_heat_maps_.cpu_data(), top[0]->mutable_cpu_data());
  // Copy the mask
  caffe_copy(top[1]->count(), this->prefetch_heat_maps_masks_.cpu_data(), top[1]->mutable_cpu_data());
}

template <typename Dtype>
void HeatMapsFromCoordsLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
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
STUB_GPU(HeatMapsFromCoordsLayer);
#endif

INSTANTIATE_CLASS(HeatMapsFromCoordsLayer);
REGISTER_LAYER_CLASS(HeatMapsFromCoords);

}  // namespace caffe