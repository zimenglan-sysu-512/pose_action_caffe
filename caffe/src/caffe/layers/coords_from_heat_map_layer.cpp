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
void CoordsFromHeatMapsLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
  // do nothing...
}


template <typename Dtype>
void CoordsFromHeatMapsLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
  const bool has_param = this->layer_param_.has_pose_coords_from_heat_maps_param();
  const bool has_bottom = bottom.size() == 2;
  CHECK(!(has_param && has_bottom)) 
      << "can't both has pose_coords_from_heat_maps_param and bottom.size() == 2...";
  if(has_param) {
    const PoseCoordsFromHeatMapsParameter pose_coords_from_heat_maps_param = 
        this->layer_param_.pose_coords_from_heat_maps_param();
    CHECK(pose_coords_from_heat_maps_param.has_heat_map_a());
    CHECK(pose_coords_from_heat_maps_param.has_heat_map_b());
    this->heat_map_a_ = pose_coords_from_heat_maps_param.heat_map_a();
    this->heat_map_b_ = pose_coords_from_heat_maps_param.heat_map_b();
  } else if(has_bottom) {
    this->heat_map_a_ = bottom[1]->cpu_data()[0];
    this->heat_map_b_ = bottom[1]->cpu_data()[1];
  } else {
    this->heat_map_a_ = 1;
    this->heat_map_b_ = 0;
  }

  this->batch_num_ = bottom[0]->num();
  this->heat_width_ = bottom[0]->width();
  this->heat_count_ = bottom[0]->count();
  this->heat_height_ = bottom[0]->height();
  this->heat_channels_ = bottom[0]->channels();
  this->key_point_num_ = this->heat_channels_;
  this->label_num_ = this->key_point_num_ * 2;
  this->heat_num_ = this->heat_width_ * this->heat_height_;

  // top blob
  top[0]->Reshape(this->batch_num_, this->label_num_, 1, 1);
  // prefetch_coordinates_labels
  this->prefetch_coordinates_labels_.Reshape(this->batch_num_, this->label_num_, 1, 1);

  if(top.size() > 1) {
    // max score of heat map for each part/joint
    CHECK_EQ(top.size(), 2);
    top[1]->Reshape(this->batch_num_, this->key_point_num_, 1, 1);
  }
  // corresponding scores
  this->prefetch_coordinates_scores_.Reshape(this->batch_num_, this->key_point_num_, 1, 1);
}

template <typename Dtype> 
void CoordsFromHeatMapsLayer<Dtype>::CreateCoordsFromHeatMap_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
  int pred_x_idx;
  int pred_y_idx;
  int max_score_idx;
  int s_offset = 0;
  int c_offset = 0;
  int hm_offset = 0;
  Dtype max_score;
  this->heat_channels_ = bottom[0]->channels();
  CHECK_LE(this->key_point_num_, this->heat_channels_);
  
  // Need to be reset or not ?
  const Dtype* heat_maps_ptr = bottom[0]->cpu_data();
  Dtype* coords_ptr = this->prefetch_coordinates_labels_.mutable_cpu_data();
  Dtype* scores_ptr = this->prefetch_coordinates_scores_.mutable_cpu_data();
  for(int item_id = 0; item_id < this->batch_num_; item_id++) {
    for(int kpn = 0; kpn < this->heat_channels_; kpn++) {
      // Initliaze
      max_score = Dtype(-FLT_MAX);
      max_score_idx = -1;
      // Deal with one part/joint
      for(int hn = 0; hn < this->heat_num_; hn++) {
        // Find max value and its corresponding index
        const Dtype score = heat_maps_ptr[hm_offset];
        if(max_score < score) {
          max_score = score;
          max_score_idx = hn;
        }
        // Increase the step/hm_offset
        hm_offset++;
      }
      CHECK_GE(max_score_idx, 0);
      CHECK_LT(max_score_idx, this->heat_num_);
      // Coordinate from heat map
      pred_x_idx = max_score_idx % this->heat_width_;
      pred_y_idx = max_score_idx / this->heat_width_;
      // Mapping: coordinate from image
      pred_x_idx = pred_x_idx * this->heat_map_a_ + this->heat_map_b_;
      pred_y_idx = pred_y_idx * this->heat_map_a_ + this->heat_map_b_;
      // Set index value (this is the initial predicted coordinates)
      coords_ptr[c_offset++] = pred_x_idx;
      coords_ptr[c_offset++] = pred_y_idx;
      // Record the corresponding sroce
      scores_ptr[s_offset++] = max_score;
    }
  }

  CHECK_EQ(hm_offset, bottom[0]->count());
  CHECK_EQ(c_offset, this->prefetch_coordinates_labels_.count());
  CHECK_EQ(s_offset, this->prefetch_coordinates_scores_.count());
}

template <typename Dtype>
void CoordsFromHeatMapsLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
  // Get coordinates
  this->CreateCoordsFromHeatMap_cpu(bottom, top);
  // Copy the preliminary&predicted coordinates labels
  caffe_copy(top[0]->count(), this->prefetch_coordinates_labels_.cpu_data(), top[0]->mutable_cpu_data());
  // Copy the corresponding maximized scores or response values
  if(top.size() == 2) {
    caffe_copy(top[1]->count(), this->prefetch_coordinates_scores_.cpu_data(), top[1]->mutable_cpu_data());
  }
}

template <typename Dtype>
void CoordsFromHeatMapsLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
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
STUB_GPU(CoordsFromHeatMapsLayer);
#endif

INSTANTIATE_CLASS(CoordsFromHeatMapsLayer);
REGISTER_LAYER_CLASS(CoordsFromHeatMaps);

}  // namespace caffe