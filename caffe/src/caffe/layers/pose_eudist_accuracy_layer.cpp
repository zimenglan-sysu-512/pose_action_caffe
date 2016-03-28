// Copyright 2015 

#include <iostream>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <cfloat>
#include <vector>
#include <string>
#include <functional>
#include <utility>
#include "boost/algorithm/string.hpp"

#include "caffe/layer.hpp"
#include "caffe/common.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/pose_tool.hpp"
#include "caffe/global_variables.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/pose_estimation_layers.hpp"

namespace caffe {

template <typename Dtype>
void PoseEuDistAccuracyLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
  CHECK(this->layer_param_.has_pose_eudist_accuracy_param());
  const PoseEuDistAccuracyParameter pudap = 
      this->layer_param_.pose_eudist_accuracy_param();
  CHECK(pudap.has_acc_factor());
  CHECK(pudap.has_acc_factor_num());
  CHECK(pudap.has_images_num());
  CHECK(pudap.has_acc_path());
  CHECK(pudap.has_acc_name());
  CHECK(pudap.has_log_name());

  // get config variables
  this->images_itemid_   = 0;
  this->images_num_      = pudap.images_num();
  this->acc_factor_      = 1; // pudap.acc_factor();
  this->acc_factor_      = acc_factor;
  this->acc_factor_num_  = pudap.acc_factor_num();
  this->acc_path_        = pudap.acc_path();
  this->acc_name_        = pudap.acc_name();
  this->log_name_        = pudap.log_name();
  this->zero_iter_test_  = false;
  if(pudap.has_zero_iter_test()) {
    this->zero_iter_test_ = pudap.zero_iter_test();
  }

  // accuracy log file
  CreateDir(this->acc_path_);
  this->acc_file_ = this->acc_path_ + this->acc_name_;
  this->log_file_ = this->acc_path_ + this->log_name_;
  LOG(INFO) << "acc_path: "   << this->acc_path_;
  LOG(INFO) << "acc_name: "   << this->acc_name_;
  LOG(INFO) << "acc_file: "   << this->acc_file_;
  LOG(INFO) << "log_file: "   << this->log_file_;
  LOG(INFO) << "acc_factor: " << this->acc_factor_;

  // threshold values
  CHECK_EQ(this->acc_factor_    , 1);
  CHECK_GT(this->acc_factor_num_, this->acc_factor_);
  CHECK_LE(this->acc_factor_num_, 100);

  // get threshold values
  LOG(INFO) << "accuracy factors below: ";
  for (int idx = 1; idx <= this->acc_factor_num_; idx++) {
    const int threshold = idx;
    
    this->max_score_.push_back(0);
    this->max_score_iter_.push_back(0);
    this->acc_factors_.push_back(threshold);

    LOG(INFO) << "idx: "       << idx << " "
              << "factor: "    << this->acc_factor_ << " "
              << "threshold: " << threshold;
  }
  // initialize
  this->label_num_     = bottom[0]->channels();
  this->key_point_num_ = this->label_num_ / 2;
  this->initAccFactors();

  LOG(INFO) << "acc_factor: "      << this->acc_factor_;
  LOG(INFO) << "acc_factors_num: " << this->acc_factor_num_;
  LOG(INFO) << "images_num: "      << this->images_num_;
  LOG(INFO) << "images_itemid: "   << this->images_itemid_;
  LOG(INFO) << "label_num: "       << this->label_num_;
  LOG(INFO) << "key_point_num: "   << this->key_point_num_;
}

template <typename Dtype>
void PoseEuDistAccuracyLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
  const std::string err_str = 
      "The data and label should have the same number.";
  CHECK_EQ(bottom[0]->num(),      bottom[1]->num()) 
      << err_str;
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels()) 
      << err_str;
  CHECK_EQ(bottom[0]->height(),   bottom[1]->height()) 
      << err_str;
  CHECK_EQ(bottom[0]->width(),    bottom[1]->width()) 
      << err_str;
  CHECK_EQ(bottom[0]->count() /   bottom[0]->num(), 
           bottom[0]->channels()) << err_str;

  // check labels' number
  this->label_num_     = bottom[0]->channels();
  this->key_point_num_ = this->label_num_ / 2;

  // check
  CHECK_GT(this->label_num_,   0);
  CHECK_EQ(this->label_num_,   this->key_point_num_ * 2);

  // diff_
  this->diff_.Reshape(bottom[0]->num(),    bottom[0]->channels(), 
                      bottom[0]->height(), bottom[0]->width());

  // top blob: record the accuracy ?
  top[0]->Reshape(1, 1, 1, 1);
}

template <typename Dtype>
void PoseEuDistAccuracyLayer<Dtype>::initAccFactors() {
	// clear
  if(!this->accuracies_.empty()) {
    for (int i = 0; i < this->acc_factor_num_; i++) {
      this->accuracies_[i].clear();
    }
    this->accuracies_.clear();
  }

	CHECK_EQ(this->accuracies_.size(), 0) 
      << "invalid accuracies variable, when reset them";

	for (int idx = 0; idx < this->acc_factor_num_; idx++) {
	  std::vector<float> acc_temp;
	  for(int sln = 0; sln < this->key_point_num_; sln++) {
	    acc_temp.push_back(0.);
	  }
    CHECK_EQ(acc_temp.size(), this->key_point_num_);
    this->accuracies_.push_back(acc_temp);
  }
  CHECK_EQ(this->accuracies_.size(), this->acc_factor_num_);
}

template <typename Dtype>
void PoseEuDistAccuracyLayer<Dtype>::InitQuantization() {
  // check
  CHECK_GE(this->images_itemid_, this->images_num_);
	// re-init
  this->initAccFactors();
  // reset images_itemid_ to be zero
  this->images_itemid_ = 0;
}

// index starts from zero 
template<typename Dtype>
void PoseEuDistAccuracyLayer<Dtype>::CalAccPerImage(
    const Dtype* pred_coords_ptr, const Dtype* gt_coords_ptr) 
{
  int index        = 0;
  const Dtype Zero = Dtype(0);
  const Dtype two  = Dtype(2.);

  Dtype pred_dist[this->key_point_num_];
  for (int idx = 0; idx < this->label_num_; idx += 2) {
    // ground true x and y
    const Dtype gt_x = gt_coords_ptr[idx + 0];
    const Dtype gt_y = gt_coords_ptr[idx + 1];

    // predicted x and y
    const Dtype pred_x  = pred_coords_ptr[idx + 0];
    const Dtype pred_y  = pred_coords_ptr[idx + 1];

    // check
    if(gt_x < Zero || gt_y < Zero || pred_x < Zero 
                   || pred_y < Zero) 
    {
      pred_dist[index++] = Dtype(this->acc_factor_num_ + 1);
      continue;
    }
    
    // distance between prediction and ground truth
    const Dtype dx      = (pred_x - gt_x) / two;
    const Dtype dy      = (pred_y - gt_y) / two;
    pred_dist[index++]  = std::sqrt(dx * dx + dy * dy);
  }
  CHECK_EQ(index, this->key_point_num_);

  // Count the correct ones
  for (int afn = 0; afn < this->acc_factor_num_; afn++) {
    const Dtype threshold = this->acc_factors_[afn];
    for (int idx = 0; idx < this->key_point_num_; idx++) {
      if (pred_dist[idx] <= threshold) {
        this->accuracies_[afn][idx]++;
      }
    }
  }
}

template <typename Dtype>
void PoseEuDistAccuracyLayer<Dtype>::WriteResults(
    const float total_accuracies[]) 
{
  std::ofstream acc_fhd;
  acc_fhd.open(this->acc_file_.c_str(), ios::out | ios::app);
  CHECK(acc_fhd);

  const int caffe_iter = GlobalVars::caffe_iter();

  acc_fhd << GlobalVars::SpiltCodeBoundWithStellate();
  acc_fhd << GlobalVars::SpiltCodeBoundWithStellate() << std::endl;
  acc_fhd << "iter: "     << GlobalVars::caffe_iter() << std::endl;
  acc_fhd << "learn_lr: " << GlobalVars::learn_lr()   << std::endl;

  for(int afn = 0; afn < this->acc_factor_num_; afn++) {
    const float acc_factor = acc_factors_[afn]; 
    const Dtype accuracy   =  Dtype(total_accuracies[afn]);

    acc_fhd << "acc_factor: " << acc_factor << " " << std::endl;
    acc_fhd << "accuracy: "   << accuracy   << std::endl;

    if(accuracy > this->max_score_[afn]) {
      this->max_score_[afn]      = accuracy;
      this->max_score_iter_[afn] = Dtype(caffe_iter);
    }
    // single
    for (int lnh = 0; lnh < this->key_point_num_ - 1; lnh++) {
      acc_fhd << lnh + 1 << "th: " << accuracies_[afn][lnh] << ", ";
    }
    acc_fhd << this->key_point_num_ << "th: " 
            << this->accuracies_[afn][this->key_point_num_ - 1];
    acc_fhd << std::endl;
  }
  
  acc_fhd << GlobalVars::SpiltCodeBoundWithStellate() << std::endl;

  for(int afn = 0; afn < this->acc_factor_num_; afn++) {
    acc_fhd << "afn: "       << afn                   << ", " 
            << "pdj: "       << acc_factors_[afn]     << ", "
            << "max score: " << this->max_score_[afn] << ", "
            << "iter: "     << this->max_score_iter_[afn] 
            << std::endl;

    LOG(INFO) << "afn: "       << afn                   << ", " 
              << "pdj: "       << acc_factors_[afn]     << ", "
              << "max score: " << this->max_score_[afn] << ", "
              << "iter: "     << this->max_score_iter_[afn];
  }
  acc_fhd << std::endl << std::endl;
  acc_fhd.close();
}

template<typename Dtype>
void PoseEuDistAccuracyLayer<Dtype>::QuanFinalResults() {
  // statistically 
  float total_accuracies[this->acc_factor_num_];

  for(int afn = 0; afn < this->acc_factor_num_; afn++) {
    total_accuracies[afn] = 0.;

    for(int lnh = 0; lnh < this->key_point_num_; lnh++) {  
      this->accuracies_[afn][lnh] /= (this->images_num_ + 0.);
      total_accuracies[afn]       += accuracies_[afn][lnh];
    }

    total_accuracies[afn] /= (this->key_point_num_ + 0.);
  }

  // save
  if(GlobalVars::caffe_iter() <= 0 && !this->zero_iter_test_) {
    return;
  }

  // write results
  this->WriteResults(total_accuracies);
}

template<typename Dtype>
void PoseEuDistAccuracyLayer<Dtype>::Quantization(
    const Dtype* pred_coords, const Dtype* gt_coords, 
    const int num) 
{
  CHECK_EQ(this->label_num_, this->key_point_num_ * 2) 
      << "invalid label_num: " << this->label_num_ 
      << ", and s_label_num: " << this->key_point_num_;

  for(int idx = 0; idx < num; idx++) {
    const Dtype* gt_coords_ptr   = gt_coords + this->label_num_   * idx;
    const Dtype* pred_coords_ptr = pred_coords + this->label_num_ * idx;

    if(this->images_itemid_ < this->images_num_) {      
      this->CalAccPerImage(pred_coords_ptr, gt_coords_ptr);
      this->images_itemid_++;
      LOG(INFO) << "images_itemid: " << this->images_itemid_ 
                << " ("              << this->images_num_ << ")";
    } else {
      LOG(INFO) << "ready for writing the accuracy results...";
      break;
    }
  }

  if(this->images_itemid_ >= this->images_num_) {
    // Record final results
    this->QuanFinalResults(); 

    // Initialize
    this->InitQuantization();
  }
}

template <typename Dtype>
void PoseEuDistAccuracyLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
	const int count = bottom[0]->count();
	const int num   = bottom[0]->num(); 
  CHECK_EQ(count, num * this->label_num_);
  
	caffe_sub(count, bottom[0]->cpu_data(), 
            bottom[1]->cpu_data(), 
            this->diff_.mutable_cpu_data());

	Dtype loss = caffe_cpu_dot(count, this->diff_.cpu_data(), 
                             this->diff_.cpu_data());

  loss /= Dtype(num + 0.);
  top[0]->mutable_cpu_data()[0] = loss;

  LOG(INFO) << "iter: " << GlobalVars::caffe_iter() 
            << ", loss: " <<  loss;

  const Dtype* gt_coords = bottom[1]->cpu_data();
  const Dtype* pred_coords = bottom[0]->cpu_data();
  this->Quantization(pred_coords, gt_coords, num);
}

INSTANTIATE_CLASS(PoseEuDistAccuracyLayer);
REGISTER_LAYER_CLASS(PoseEuDistAccuracy);

}  // namespace caffe