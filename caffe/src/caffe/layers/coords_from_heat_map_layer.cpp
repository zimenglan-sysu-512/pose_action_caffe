// Copyright 2015 DDK (dongdk.sysu@foxmail.com)

#include <algorithm>
#include <cfloat>
#include <vector>
#include <map>

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

struct partfilternode {
  int heatmapindice;
  string parent;
  int xmove;
  int ymove;
  int width;
  int height;
};

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
    this->topK_       = pose_coords_from_heat_maps_param.top_k();
  } else if(has_bottom) {
    this->heat_map_a_ = bottom[1]->cpu_data()[0];
    this->heat_map_b_ = bottom[1]->cpu_data()[1];
  } else {
    this->heat_map_a_ = 1;
    this->heat_map_b_ = 0;
    this->topK_       = 0;
  }
  this->batch_num_     = bottom[0]->num();
  this->heat_width_    = bottom[0]->width();
  this->heat_count_    = bottom[0]->count();
  this->heat_height_   = bottom[0]->height();
  this->heat_channels_ = bottom[0]->channels();
  this->key_point_num_ = this->heat_channels_;
  this->label_num_     = this->key_point_num_ * 2;
  this->heat_num_      = this->heat_width_ * this->heat_height_;

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

  if(this->layer_param_.is_disp_info()) {
    LOG(INFO) << "heat_map_a: " << this->heat_map_a_;
    LOG(INFO) << "heat_map_b: " << this->heat_map_b_;
    LOG(INFO) << "topK: "       << this->topK_;
  }
}

template <typename Dtype>
void CoordsFromHeatMapsLayer<Dtype>::FilterHeatMap(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{ /* Written by Shengfu Zhai */
  Dtype* heatmapptr = bottom[0]->mutable_cpu_data();
  int num       = bottom[0]->num();
  int channels  = bottom[0]->channels();
  int height    = bottom[0]->height();
  int width     = bottom[0]->width();
  int dimension = height * width;
  std::map<string,partfilternode> partfilterconf;
  partfilterconf.insert(std::pair<string, partfilternode>("03head", partfilternode{0,"02neck",0,0,2,2}));
  partfilterconf.insert(std::pair<string, partfilternode>("02neck",  partfilternode{1,"01neckbelow",0,0,2,2}));
  partfilterconf.insert(std::pair<string, partfilternode>("01neckbelow",partfilternode{2,"00torso",0,-2,2,2}));
  partfilterconf.insert(std::pair<string, partfilternode>("00torso",partfilternode{3,"root",0,0,0,0}));
  partfilterconf.insert(std::pair<string, partfilternode>("12jj",partfilternode{4,"00torso",0,2,2,2}));
  partfilterconf.insert(std::pair<string, partfilternode>("04rightshoulder",partfilternode{5,"00torso",-1,-2,2,2}));
  partfilterconf.insert(std::pair<string, partfilternode>("05rightelbow",partfilternode{6,"04rightshoulder",-1,0,3,3}));
  partfilterconf.insert(std::pair<string, partfilternode>("06rightankel",partfilternode{7,"05rightelbow",0,0,3,3}));
  partfilterconf.insert(std::pair<string, partfilternode>("07righthand",partfilternode{8,"06rightankel",0,0,1,1}));
  partfilterconf.insert(std::pair<string, partfilternode>("08leftshoulder",partfilternode{9,"00torso",1,-1,2,2}));
  partfilterconf.insert(std::pair<string, partfilternode>("09leftelbow",partfilternode{10,"08leftshoulder",0,0,3,3}));
  partfilterconf.insert(std::pair<string, partfilternode>("10leftankel",partfilternode{11,"09leftelbow",0,0,3,3}));
  partfilterconf.insert(std::pair<string, partfilternode>("11lefthand",partfilternode{12,"10leftankel",0,0,1,1}));
  partfilterconf.insert(std::pair<string, partfilternode>("13righthip",partfilternode{13,"00torso",0,3,2,2}));
  partfilterconf.insert(std::pair<string, partfilternode>("14rightknee",partfilternode{14,"13righthip",0,4,3,3}));
  partfilterconf.insert(std::pair<string, partfilternode>("15rightfoot",partfilternode{15,"14rightknee",0,4,2,2}));
  partfilterconf.insert(std::pair<string, partfilternode>("16lefthip",partfilternode{16,"00torso",0,3,2,2}));
  partfilterconf.insert(std::pair<string, partfilternode>("17leftknee",partfilternode{17,"16lefthip",1,4,3,3}));
  partfilterconf.insert(std::pair<string, partfilternode>("18leftfoot",partfilternode{18,"17leftknee",1,4,2,2}));
  std::map<string,partfilternode>::iterator  tem;
  for(tem = partfilterconf.begin(); tem!= partfilterconf.end(); tem++){
    if(tem->first == "00torso"){
      continue;
    }
    Dtype filtermask[height][width];
    Dtype* currentheatmap = &heatmapptr[  tem->second.heatmapindice * width * height];
    Dtype* parentheatmap = &heatmapptr[ ( partfilterconf.find(tem->second.parent)->second.heatmapindice )*width*height];
    Dtype maxtem = Dtype(-FLT_MAX);
    int x=0,y=0;
    for(int d=0; d<dimension; d++){
      if(parentheatmap[d] > maxtem){
        maxtem = parentheatmap[d];
        y = d;
      }
      x++;
    }
    x = y  % width;
    y = y / width;
    int xmove = tem->second.xmove;
    int ymove = tem->second.ymove;
    int halfwidth = tem->second.width;
    int halfheight = tem->second.height;
    x = x + xmove;
    y = y + ymove;
    for(int i =0; i < height; i++){
      for (int j =0;j< width; j++){
        if(j< x - halfwidth || j > x + halfwidth || i < y - halfheight || i > y+ halfheight){
          filtermask[i][j] = 0.0;
        }
        else{
          filtermask[i][j] = 10.0;
        }
      }
    }
    caffe_mul<Dtype>( height*width , currentheatmap, (Dtype*)filtermask,currentheatmap );
  }
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
void CoordsFromHeatMapsLayer<Dtype>::CreateCoordsFromHeatMapFromTopK_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
  int topK;
  int pred_x_idx;
  int pred_y_idx;
  int s_offset  = 0;
  int c_offset  = 0;
  int hm_offset = 0;
  Dtype max_score;
  this->heat_channels_ = bottom[0]->channels();
  CHECK_LE(this->key_point_num_, this->heat_channels_);
  
  // Need to be reset or not ?
  const Dtype* heat_maps_ptr = bottom[0]->cpu_data();
  Dtype* coords_ptr          = this->prefetch_coordinates_labels_.mutable_cpu_data();
  Dtype* scores_ptr          = this->prefetch_coordinates_scores_.mutable_cpu_data();

  for(int item_id = 0; item_id < this->batch_num_; item_id++) {
    for(int kpn = 0; kpn < this->heat_channels_; kpn++) {
      std::vector<std::pair<Dtype, int> > tops; // <score, index>
      for(int hn = 0; hn < this->heat_num_; hn++) {
        tops.push_back(std::make_pair(heat_maps_ptr[hm_offset], hn));
        hm_offset++;
      }
      // sort
      topK = tops.size();
      topK = std::min(this->topK_, topK);
      CHECK_GT(topK, 0);
      std::partial_sort(tops.begin(), tops.begin() + topK, tops.end(), 
          std::greater<pair<Dtype, int> >());

      // find corresponding coordinates
      pred_x_idx = 0;
      pred_y_idx = 0;
      max_score  = Dtype(0);
      for(int k = 0; k < topK; k++) {
        max_score  += tops[k].first;
        pred_x_idx += (tops[k].second % this->heat_width_);
        pred_y_idx += (tops[k].second / this->heat_width_);
      }
      max_score  /= Dtype(topK);
      pred_x_idx /= topK;
      pred_y_idx /= topK;

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
  if(this->topK_ > 0) {
    if(this->layer_param_.is_disp_info()) {
      LOG(INFO) << "Using top K highest scores for part location";
    }
    this->CreateCoordsFromHeatMapFromTopK_cpu(bottom, top);
  } else {
    if(this->layer_param_.is_disp_info()) {
      LOG(INFO) << "Using highest scores for part location";
    }
    this->CreateCoordsFromHeatMap_cpu(bottom, top);
  }

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