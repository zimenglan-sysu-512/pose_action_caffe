// Copyright 2015 DDK (dongdk.sysu@foxmail.com)

#include <algorithm>
#include <cfloat>
#include <vector>
#include "boost/algorithm/string.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/core/core.hpp>

#include "caffe/pose_estimation_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/common.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/pose_tool.hpp"
#include "caffe/global_variables.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

using std::min;
using std::max;

template <typename Dtype>
void HeatMapsFromCoordsOrigin2Layer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
  CHECK(this->layer_param_.has_heat_maps_from_coords_origin2_param());
  const HeatMapsFromCoordsOrigin2Parameter heat_maps_from_coords_origin2_param = 
      this->layer_param_.heat_maps_from_coords_origin2_param();

  CHECK(heat_maps_from_coords_origin2_param.has_is_binary());
  CHECK(heat_maps_from_coords_origin2_param.has_gau_mean());
  CHECK(heat_maps_from_coords_origin2_param.has_gau_stds());

  this->is_binary_ = heat_maps_from_coords_origin2_param.is_binary(); 
  this->gau_mean_  = heat_maps_from_coords_origin2_param.gau_mean();
  this->gau_stds_  = heat_maps_from_coords_origin2_param.gau_stds();
  CHECK(this->gau_stds_ > 0);

  const bool has_radius = heat_maps_from_coords_origin2_param.has_radius();
  const bool has_radius_str = heat_maps_from_coords_origin2_param.has_radius_str();
  CHECK((!has_radius && has_radius_str) || (has_radius && !has_radius_str));
  if(has_radius || has_radius_str) {
    this->batch_num_ = bottom[0]->num();
    this->label_num_ = bottom[0]->count() / this->batch_num_;
    this->key_point_num_ = this->label_num_ / 2;
    CHECK_EQ(this->label_num_, this->key_point_num_ * 2);
    CHECK_EQ(this->label_num_, bottom[0]->channels());

    if(has_radius) {
      this->radius_  = heat_maps_from_coords_origin2_param.radius();
      CHECK_GT(this->radius_, 0);
      for(int idx = 0; idx < this->key_point_num_; idx++) {
        this->all_radius_.push_back(this->radius_);
      }
    } else if(has_radius_str){
      const std::string radius_str = heat_maps_from_coords_origin2_param.radius_str();
      vector<string> radius_str_s;
      boost::split(radius_str_s, radius_str, boost::is_any_of(","));
      int rss = radius_str_s.size();
      CHECK_EQ(rss, this->key_point_num_);
      for(int idx = 0; idx < this->key_point_num_; idx++) {
        this->all_radius_.push_back(std::atoi(radius_str_s[idx].c_str()));
      }
    }
  } else {
    NOT_IMPLEMENTED;
  }

  const bool has_in_radius = heat_maps_from_coords_origin2_param.has_in_radius();
  const bool has_in_radius_str = heat_maps_from_coords_origin2_param.has_in_radius_str();
  CHECK((!has_in_radius && has_in_radius_str) || (has_in_radius && !has_in_radius_str));
  if(has_in_radius || has_in_radius_str) {
    if(has_in_radius) {
      this->in_radius_  = heat_maps_from_coords_origin2_param.in_radius();
      CHECK_GT(this->in_radius_, 0);
      for(int idx = 0; idx < this->key_point_num_; idx++) {
        this->all_in_radius_.push_back(this->in_radius_);
      }
    } else if(has_in_radius_str){
      const std::string in_radius_str = heat_maps_from_coords_origin2_param.in_radius_str();
      vector<string> in_radius_str_s;
      boost::split(in_radius_str_s, in_radius_str, boost::is_any_of(","));
      int rss = in_radius_str_s.size();
      CHECK_EQ(rss, this->key_point_num_);
      for(int idx = 0; idx < this->key_point_num_; idx++) {
        this->all_in_radius_.push_back(std::atoi(in_radius_str_s[idx].c_str()));
      }
    }
  } else {
    NOT_IMPLEMENTED;
  }
  CHECK_EQ(this->all_radius_.size(), this->all_in_radius_.size());
  for(int s = 0; s < this->all_radius_.size(); s++) {
    CHECK_GT(this->all_in_radius_[s], this->all_radius_[s]);
  }

  this->has_visual_path_ = false;
  if(heat_maps_from_coords_origin2_param.has_visual_path()) {
    this->has_visual_path_ = true;
  }
  if(this->has_visual_path_) {
    this->visual_path_ = heat_maps_from_coords_origin2_param.visual_path();  
    CreateDir(this->visual_path_.c_str(), 0);  
    CHECK(heat_maps_from_coords_origin2_param.has_img_ext()); 
    this->img_ext_ = heat_maps_from_coords_origin2_param.img_ext();
    this->mask_visual_path_ = this->visual_path_ + "masks/";
    CreateDir(this->mask_visual_path_.c_str(), 0); 
    this->label_visual_path_ = this->visual_path_ + "labels/";
    CreateDir(this->label_visual_path_.c_str(), 0); 
  }

  LOG(INFO) << "*************************************************";
  LOG(INFO) << "mean: " << this->gau_mean_;
  LOG(INFO) << "stds: " << this->gau_stds_;
  LOG(INFO) << "is_binary: " << this->is_binary_;
  LOG(INFO) << "radius: ";
  for(int idx = 0; idx < this->key_point_num_; idx++) {
    LOG(INFO) << "id: " << idx << " -- radius: " << this->all_radius_[idx];
  }
  for(int idx = 0; idx < this->key_point_num_; idx++) {
    LOG(INFO) << "id: " << idx << " -- in_radius: " << this->all_in_radius_[idx];
  }
  if(this->has_visual_path_) {
    LOG(INFO) << "visual_path: " << this->visual_path_; 
    LOG(INFO) << "label_visual_path: " << this->label_visual_path_; 
    LOG(INFO) << "mask_visual_path: " << this->mask_visual_path_; 
  }
  LOG(INFO) << "**************************************************";
}

template <typename Dtype>
void HeatMapsFromCoordsOrigin2Layer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
  this->batch_num_ = bottom[0]->num();
  this->heat_channels_ = bottom[0]->channels();
  this->label_num_ = bottom[0]->count() / this->batch_num_;
  this->key_point_num_ = this->label_num_ / 2;
  CHECK_EQ(this->label_num_, this->key_point_num_ * 2);
  CHECK_EQ(this->label_num_, bottom[0]->channels());
  CHECK_EQ(this->key_point_num_, this->all_radius_.size());
  CHECK_EQ(this->heat_channels_, this->key_point_num_ * 2);

  /// aux info (img_ind, width, height, im_scale, flippable)
  CHECK_EQ(bottom[1]->num(), bottom[0]->num());
  CHECK_EQ(bottom[1]->channels(), 5);
  CHECK_EQ(bottom[1]->channels(), bottom[1]->count() / bottom[1]->num());

  const Dtype* aux_info = bottom[1]->cpu_data();
  Dtype max_width = Dtype(-1);
  Dtype max_height = Dtype(-1);
  for(int item_id = 0; item_id < this->batch_num_; item_id++) {
    /// (img_ind, width, height, im_scale, flippable)
    const int offset = bottom[1]->offset(item_id);
    /// const Dtype img_ind   = aux_info[offset + 0];
    const Dtype width     = aux_info[offset + 1];
    const Dtype height    = aux_info[offset + 2];
    const Dtype im_scale  = aux_info[offset + 3];
    /// const Dtype flippable = aux_info[offset + 4];
    max_width = std::max(max_width, width * im_scale);
    max_height = std::max(max_height, height * im_scale);
  }
  // reshape & reset
  this->heat_width_ = int(max_width);
  this->heat_height_ = int(max_height);
  this->heat_num_ = this->heat_width_ * this->heat_height_;

  // top blob
  top[0]->Reshape(this->batch_num_, this->key_point_num_, this->heat_height_, this->heat_width_);
  // prefetch_coordinates_labels
  this->prefetch_heat_maps_.Reshape(this->batch_num_, this->key_point_num_, this->heat_height_, this->heat_width_);
  // mask
  top[1]->Reshape(this->batch_num_, this->key_point_num_, this->heat_height_, this->heat_width_);
  this->prefetch_heat_maps_masks_.Reshape(this->batch_num_, this->key_point_num_, this->heat_height_, this->heat_width_);

  // (heat_width, heat_height, heat_map_a, heat_map_b)
  if(top.size() == 3) {
    top[2]->Reshape(4, 1, 1, 1);
    top[2]->mutable_cpu_data()[0] = Dtype(this->heat_width_);
    top[2]->mutable_cpu_data()[1] = Dtype(this->heat_height_);
    top[2]->mutable_cpu_data()[2] = Dtype(1);
    top[2]->mutable_cpu_data()[3] = Dtype(0);
  }
}

template <typename Dtype> 
void HeatMapsFromCoordsOrigin2Layer<Dtype>::CreateHeatMapsFromCoordsOrigin2(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
  const Dtype One = Dtype(1);
  const Dtype Zero = Dtype(0);
  const Dtype Stds = this->gau_stds_ * this->gau_stds_ * Dtype(-2);
  
  const Dtype* coords_ptr = bottom[0]->cpu_data();
  Dtype* heat_maps_ptr = this->prefetch_heat_maps_.mutable_cpu_data();
  Dtype* heat_maps_masks_ptr = this->prefetch_heat_maps_masks_.mutable_cpu_data();
  caffe_set(this->prefetch_heat_maps_.count(), Zero, heat_maps_ptr);
  caffe_set(this->prefetch_heat_maps_masks_.count(), One, heat_maps_masks_ptr);

  // start
  int kpn_count;
  const Dtype* aux_info = bottom[1]->cpu_data();
  for(int bn = 0; bn < this->batch_num_; bn++) {
    // get offset of coordinates
    const int coords_offset = bottom[0]->offset(bn);
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
      const Dtype ox = coords_ptr[coords_offset + idx + 0];
      const Dtype oy = coords_ptr[coords_offset + idx + 1];
      // invalid coordinates
      if(ox < Zero || oy < Zero || ox >= r_width|| oy >= r_height) { 
        const int heat_map_masks_offset = this->prefetch_heat_maps_masks_.offset(bn, kpn);
        caffe_set(this->heat_num_, Zero, heat_maps_masks_ptr + heat_map_masks_offset);
        continue; 
      }

      // heat map
      const int radius = this->all_radius_[kpn];
      const Dtype radius_area = radius * radius;
      for(int dx = -radius; dx <= radius; dx++) {
        const Dtype hx = ox + dx;
        if(hx < Zero || hx >= r_width) { continue; }
        for(int dy = -radius; dy <= radius; dy++) {
          const Dtype hy = oy + dy;
          if(hy < Zero || hy >= r_height) { continue; }
          const Dtype square = dx * dx + dy * dy;
          if(square > radius_area) { continue; }
          kpn_count++;

          // get offset for heat maps
          const int index = this->prefetch_heat_maps_.offset(bn, kpn, int(hy), int(hx)); 
          if(this->is_binary_) {
            heat_maps_ptr[index] = One;
          } else {
            // mean_x = mean_y = 0, std_x = std_y = 1.5
            // for simplicity, f(x, y) = exp(-4.5 * (x^2 + y^2))
            const Dtype square2 = (dx - this->gau_mean_) * (dx - this->gau_mean_) + 
                (dy - this->gau_mean_) * (dy - this->gau_mean_);
            const Dtype dist = exp(- square2 / Stds);
            heat_maps_ptr[index] = max(heat_maps_ptr[index], dist);
          }
        }
      }

      // heat map mask
      if(kpn_count <= 0) {
        const int heat_map_masks_offset = this->prefetch_heat_maps_masks_.offset(bn, kpn);
        caffe_set(this->heat_num_, Zero, heat_maps_masks_ptr + heat_map_masks_offset);
        continue;
      }
      const int in_radius = this->all_in_radius_[kpn];
      const Dtype in_radius_area = in_radius * in_radius;
      for(int dx = -in_radius; dx <= in_radius; dx++) {
        const Dtype hx = ox + dx;
        if(hx < Zero || hx >= r_width) { continue; }
        for(int dy = -in_radius; dy <= in_radius; dy++) {
          const Dtype hy = oy + dy;
          if(hy < Zero || hy >= r_height) { continue; }
          const Dtype square = dx * dx + dy * dy;
          if(square <= radius_area) { continue; }
          if(square >= in_radius_area) { continue; }

          // get offset for heat maps
          const int index = this->prefetch_heat_maps_masks_.offset(bn, kpn, int(hy), int(hx));
          heat_maps_masks_ptr[index] = Zero;
        }
      }
    } // end kpn
  } // end bn
}

template <typename Dtype>
void HeatMapsFromCoordsOrigin2Layer<Dtype>::Visualize() {
  const std::vector<std::string>& objidxs = GlobalVars::objidxs();
  const std::vector<std::string>& imgidxs = GlobalVars::imgidxs();
  CHECK_EQ(objidxs.size(), imgidxs.size());
  CHECK_EQ(objidxs.size(), this->batch_num_);
  
  const Dtype* heat_maps = this->prefetch_heat_maps_.cpu_data();
  for(int n = 0; n < this->batch_num_; n++) {
    for(int c = 0; c < this->key_point_num_; c++) {
      cv::Mat img = cv::Mat::zeros(this->heat_height_, this->heat_width_, CV_8UC1);
      for(int h = 0; h < this->heat_height_; h++) {
        for(int w = 0; w < this->heat_width_; w++) {
          // get offset
          const int offset = this->prefetch_heat_maps_.offset(n, c, h, w);
          // (rows, cols) <---> (height, width)
          img.at<uchar>(h, w) = heat_maps[offset];
          // LOG(INFO) << heat_maps[offset];
        }
      }
      // 
      const std::string img_path = this->label_visual_path_ + imgidxs[n] + "_" 
          + objidxs[n] + "_" + to_string(c) + this->img_ext_;
      cv::imwrite(img_path, img);
    }
  }
  // LOG(INFO) << "visualize heat maps done!";
  const Dtype* masks = this->prefetch_heat_maps_masks_.cpu_data();
  for(int n = 0; n < this->batch_num_; n++) {
    for(int c = 0; c < this->key_point_num_; c++) {
      cv::Mat img = cv::Mat::zeros(this->heat_height_, this->heat_width_, CV_8UC1);
      for(int h = 0; h < this->heat_height_; h++) {
        for(int w = 0; w < this->heat_width_; w++) {
          // get offset
          const int offset = this->prefetch_heat_maps_masks_.offset(n, c, h, w);
          // (rows, cols) <---> (height, width)
          img.at<uchar>(h, w) = masks[offset];
        }
      }
      // 
      const std::string img_path = this->mask_visual_path_ + imgidxs[n] + "_" 
          + objidxs[n] + "_" + to_string(c) + this->img_ext_;
      cv::imwrite(img_path, img);
    }
  }
  // LOG(INFO) << "visualize heat map masks done!";
}

template <typename Dtype>
void HeatMapsFromCoordsOrigin2Layer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
  // Get Heat maps for batch
  CreateHeatMapsFromCoordsOrigin2(bottom, top);
  // Copy the heat maps
  caffe_copy(top[0]->count(), this->prefetch_heat_maps_.cpu_data(), top[0]->mutable_cpu_data());
  // Copy the mask
  caffe_copy(top[1]->count(), this->prefetch_heat_maps_masks_.cpu_data(), top[1]->mutable_cpu_data());

  // visualization
  if(this->has_visual_path_) {
    Visualize();
  }
}

template <typename Dtype>
void HeatMapsFromCoordsOrigin2Layer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
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
STUB_GPU(HeatMapsFromCoordsOrigin2Layer);
#endif

INSTANTIATE_CLASS(HeatMapsFromCoordsOrigin2Layer);
REGISTER_LAYER_CLASS(HeatMapsFromCoordsOrigin2);

}  // namespace caffe