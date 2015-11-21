// Copyright DDK 2015 

#include <iostream>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <cfloat>
#include <vector>
// #include <pair>
#include <map>
#include <string>

#include "caffe/pose_estimation_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/common.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/global_variables.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"


namespace caffe {

template <typename Dtype>
void PoseHeatMapLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
  LossLayer<Dtype>::LayerSetUp(bottom, top);

  CHECK(this->layer_param_.has_pose_heat_map_loss_param());
  const PoseHeatMapLossParameter pose_heat_map_loss_param = 
      this->layer_param_.pose_heat_map_loss_param();
  CHECK(pose_heat_map_loss_param.has_fg_eof());
  CHECK(pose_heat_map_loss_param.has_bg_eof());
  CHECK(pose_heat_map_loss_param.has_loss_emphase_type());
  this->has_ratio_ = pose_heat_map_loss_param.has_ratio();
  this->has_bg_num_ = pose_heat_map_loss_param.has_bg_num();
  CHECK((!this->has_ratio_ && this->has_bg_num_) 
      || (this->has_ratio_ && !this->has_bg_num_));

  this->fg_eof_ = pose_heat_map_loss_param.fg_eof();
  this->bg_eof_ = pose_heat_map_loss_param.bg_eof();
  this->loss_emphase_type_ = pose_heat_map_loss_param.loss_emphase_type();
  if(this->has_ratio_) {
    this->ratio_  = pose_heat_map_loss_param.ratio();
  } else if(this->has_bg_num_) {
    this->bg_num_ = pose_heat_map_loss_param.bg_num();
    CHECK_GT(this->bg_num_, 0);
    const int heat_num = bottom[0]->width() * bottom[0]->height();
    this->ratio_ = Dtype(this->bg_num_) / Dtype(heat_num + 0.);
  } else {
    NOT_IMPLEMENTED;
  }

  CHECK(pose_heat_map_loss_param.has_prob_num());
  CHECK(pose_heat_map_loss_param.has_heat_score_thres());
  CHECK(pose_heat_map_loss_param.has_parts_err_num_thres());
  this->prob_num_ = pose_heat_map_loss_param.prob_num();
  this->parts_err_num_thres_ = pose_heat_map_loss_param.parts_err_num_thres();
  this->heat_score_thres_ = pose_heat_map_loss_param.heat_score_thres();
  // this->heat_score_thres_arr_ = pose_heat_map_loss_param.heat_score_thres_arr();
  this->hard_negative_filepath_ = pose_heat_map_loss_param.hard_negative_filepath();

  CHECK_GT(this->ratio_, 0.);
  CHECK_LE(this->ratio_, 1.);
  CHECK(this->prob_num_ > 0);
  // UINT_MAX(unsigned int): 4294967295, uint_thres_: 42949672
  this->uint_thres_ = static_cast<unsigned int>(UINT_MAX * this->ratio_);
  LOG(INFO) << "loss_emphase_type: " << this->loss_emphase_type_;
  LOG(INFO) << "fg_eof: " << this->fg_eof_;
  LOG(INFO) << "bg_eof: " << this->bg_eof_;
  LOG(INFO) << "ratio: " << this->ratio_;
  LOG(INFO) << "prob_num: " << this->prob_num_;
  LOG(INFO) << "parts_err_num_thres: " << this->parts_err_num_thres_;
  LOG(INFO) << "heat_score_thres: " << this->heat_score_thres_;
  LOG(INFO) << "hard_negative_filepath: " << this->hard_negative_filepath_;
  LOG(INFO);
  this->InitRand();
}

template <typename Dtype>
void PoseHeatMapLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
  // call the parent-class function
  LossLayer<Dtype>::Reshape(bottom, top);
  // heat maps
  CHECK_EQ(bottom[0]->count(), bottom[1]->count())
      << "Inputs must have the same dimension.";
  CHECK_EQ(bottom[0]->num(), bottom[1]->num())
      << "The data and label should have the same number.";
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
  CHECK_EQ(bottom[0]->height(), bottom[1]->height());
  CHECK_EQ(bottom[0]->width(), bottom[1]->width());

  // heat map mask: indicate which heat map is valid or needs bp
  if(bottom.size() == 3) {
    CHECK_EQ(bottom[0]->num(), bottom[2]->num())
        << "The data and label should have the same number.";
    CHECK_EQ(bottom[0]->channels(), bottom[2]->channels())
        << "Inputs must have the same dimension.";  
    CHECK_EQ(bottom[2]->channels(), bottom[2]->count() / bottom[2]->num());
  }

  // since the heat map can dynamically changed, 
  // so be cafeful use `heat_num_`(etc) variables
  this->key_point_num_ = bottom[0]->channels();
  CHECK_GT(this->key_point_num_, 0);
  this->heat_num_ = bottom[0]->width() * bottom[0]->height();
  if(this->has_bg_num_) {
    this->ratio_ = Dtype(this->bg_num_) / Dtype(this->heat_num_ + 0.);
  }

  // Reference to `src/caffe/layers/dropout_layer.cpp` or `src/caffe/neuron_layers.hpp`
  rand_vec_.Reshape(1, 1, bottom[0]->height(), bottom[0]->width());
  this->diff_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void PoseHeatMapLossLayer<Dtype>::InitRand() {
  // here set true forever
  const bool needs_rand = true;
  if (needs_rand) {
    const unsigned int rng_seed = caffe_rng_rand();
    rng_.reset(new Caffe::RNG(rng_seed));
  } else {
    rng_.reset();
  }
}

template <typename Dtype>
int PoseHeatMapLossLayer<Dtype>::Rand(int n) {
  CHECK(rng_);
  CHECK_GT(n, 0);
  caffe::rng_t* rng = static_cast<caffe::rng_t*>(rng_->generator());
  return ((*rng)() % n);
}

template <typename Dtype>
void PoseHeatMapLossLayer<Dtype>::ComputesHeatMapLoss(
    const vector<Blob<Dtype>*>& bottom) 
{
  const int num = bottom[0]->num();
  const int count = bottom[0]->count();
  const int heat_map_channels = bottom[0]->channels();
  CHECK_LE(this->key_point_num_, heat_map_channels);

  /* Get data&labels pointers */
  const Dtype* pred_data = bottom[0]->cpu_data();
  const Dtype* gt_data = bottom[1]->cpu_data();
  Dtype* diff_data = this->diff_.mutable_cpu_data();

  // Dtype x, y;
  const Dtype Zero = Dtype(0);
  const Dtype kErrorBound = Dtype(1e-4);  

  /* Default */
  if (this->loss_emphase_type_ <= 0) {
    for(int idx = 0; idx < count; idx++) {
      diff_data[idx] = pred_data[idx] - gt_data[idx];
    }

  /* Consider only the ground truth */
  } else if(this->loss_emphase_type_ == 1) {
    for(int idx = 0; idx < count; idx++) {
      diff_data[idx] = pred_data[idx] - gt_data[idx];

      if(gt_data[idx] <= (Zero + kErrorBound)) {
        diff_data[idx] = Dtype(0);
      }
    }

  /* Consider backgrounds and ground truths(but scale ground truths) */
  } else if(this->loss_emphase_type_ == 2) {
    for(int idx = 0; idx < count; idx++) {
      // compute the diff
      diff_data[idx] = pred_data[idx] - gt_data[idx];

      if(gt_data[idx] > (Zero + kErrorBound)) {
        diff_data[idx] *= this->fg_eof_;
      }
    }

  /* Consider a littel bit of  backgrounds and ground truths(but scale them) */
  } else if(this->loss_emphase_type_ == 3 || this->loss_emphase_type_ == 4) {
    // get foreground eof 
    const Dtype fg_eof2 = this->loss_emphase_type_ == 
        4 ? this->fg_eof_ : Dtype(1);
    // produce the random number
    unsigned int* rand_mask = rand_vec_.mutable_cpu_data();

    // get offset
    int hm_off = 0;
    for(int np = 0; np < num; np++) {
      for(int kpn = 0; kpn < heat_map_channels; kpn++) {
        /* Create random numbers */
        caffe_rng_bernoulli(this->heat_num_, this->ratio_, rand_mask);

        for(int hn = 0; hn < this->heat_num_; hn++) {
          // conditions
          const bool flag1 = rand_mask[hn] > 0;
          const bool flag2 = gt_data[hm_off] > (Zero + kErrorBound);
          const bool flag3 = pred_data[hm_off] >= this->heat_score_thres_;
          // compute diff
          diff_data[hm_off] = pred_data[hm_off] - gt_data[hm_off];
          // fg
          if(flag2) {
            diff_data[hm_off] *= fg_eof2;  
          // bg
          } else if(flag1) {
            diff_data[hm_off] *= this->bg_eof_;
          // hard negative -- randomly set zero
          } else if(flag3) {
            if(this->Rand(this->prob_num_)) {
              diff_data[hm_off] = Dtype(0);      
            }
          // the backgrounds either keep the same 
          } else {
            diff_data[hm_off] = Dtype(0);
          }

          // increase the offset by one
          hm_off++;
        }
      }
    }
    CHECK_EQ(hm_off, count) << "does not match the size of each blob in bottom";
  
  /* Consider a littel bit of  backgrounds and ground truths(but scale them) */
  } else if(this->loss_emphase_type_ == 5) {
    // get offset
    int hm_off = 0;
    int l1 = 0, len = 0;
    int l2 = int(this->heat_num_* this->ratio_);
    // force to be 1
    this->fg_eof_ = 1;
    // this->fg_eof_ = 1.12;
    // force to multiply 2
    // l2 = l2 * 2;
    // produce the random number
    unsigned int* rand_mask = rand_vec_.mutable_cpu_data();

    for(int np = 0; np < num; np++) {
      for(int kpn = 0; kpn < heat_map_channels; kpn++) {
        /* Create random numbers */
        caffe_rng_bernoulli(this->heat_num_, this->ratio_, rand_mask);
        // init map to record hard negatives
        std::vector<std::pair<int, Dtype> > hards;
        for(int hn = 0; hn < this->heat_num_; hn++) {
          // condition
          const bool flag1 = rand_mask[hn] > 0;
          const bool flag2 = gt_data[hm_off] > (Zero + kErrorBound);
          const bool flag3 = pred_data[hm_off] >= this->heat_score_thres_;
          // compute diff
          diff_data[hm_off] = pred_data[hm_off] - gt_data[hm_off];
          // fg
          if(flag2) {
            diff_data[hm_off] *= this->fg_eof_;
          // bg -- randomly
          } else if(flag1) {
            diff_data[hm_off] *= this->bg_eof_;
          // hard negatives -- record
          } else if(flag3) {
            hards.push_back(std::make_pair(hm_off, diff_data[hm_off]));
          // otherwise -- set zero
          } else {
            diff_data[hm_off] = Dtype(0); 
          }
          // increase the offset by one
          hm_off++;
        } // end hn
        // sort
        l1 = hards.size();
        len = std::min(l1, l2);
        // descending order -- see statci member `cmp` function
        std::partial_sort(hards.begin(), hards.begin() + len, hards.end(), PoseHeatMapLossLayer::cmp);
        for(int i = 0; i < len; i++) {
          diff_data[hards[i].first] = hards[i].second;
          // LOG(INFO) << "np: " << np << ", kpn: " << kpn
          //     << ", l1: " << l1 << ", l2: " << l2
          //     << ", len: " << len << ", f: " << hards[i].first
          //     << ", s: " << hards[i].second;
        }
        // rest of unordered hard negatives
        // dose not need shuffle, since Rand func will work as shuffle.
        for(int j = len; j < hards.size(); j++) {
          // only retain 1/prob_num of hard negative
          if(this->Rand(this->prob_num_)) {
            diff_data[hards[j].first] = Dtype(0);
          }
        }


      } // end kpn
    } // end np
    CHECK_EQ(hm_off, count) << "does not match the size of each blob in bottom";
  
  } else {
    if(this->loss_emphase_type_) {
      LOG(INFO) << "loss_emphase_type_: " << this->loss_emphase_type_;
      NOT_IMPLEMENTED;
    }
  }
}

template <typename Dtype>
Dtype PoseHeatMapLossLayer<Dtype>::PrintLoss() 
{
  Dtype batch_loss = caffe_cpu_dot(
      this->diff_.count(), 
      this->diff_.cpu_data(), 
      this->diff_.cpu_data()
  );
  const Dtype per_frame_loss = batch_loss / Dtype(this->diff_.num() + 0.);
  const Dtype per_heat_map_loss = per_frame_loss / Dtype(this->key_point_num_ + 0.);

  LOG(INFO) << "iter: " << GlobalVars::caffe_iter() 
    << ", learn_lr: " << GlobalVars::learn_lr();
  LOG(INFO) << "euclidean loss (batch): " << batch_loss ;
  LOG(INFO) << "euclidean loss (frame): " << per_frame_loss;
  LOG(INFO) << "euclidean loss (joint): " << per_heat_map_loss;
  LOG(INFO) << "loss_emphase_type: " << loss_emphase_type_;
  LOG(INFO) << "heat_score_thres: " << heat_score_thres_;
  LOG(INFO) << "prob_num: " << this->prob_num_;
  LOG(INFO);
  
  return batch_loss;
}

/**
 * @brief bottom[0] is predicted blob, bottom[1] is ground truth blob
 * But sometimes, bottom[1] may be another predicted blob (as in siamese network)
 * So, ...
*/
template <typename Dtype>
void PoseHeatMapLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
  // Compute correspoding with loss_emphase_type
  this->ComputesHeatMapLoss(bottom);
  // Print loss
  Dtype loss = this->PrintLoss();
  top[0]->mutable_cpu_data()[0] = loss;
}


template <typename Dtype>
void PoseHeatMapLossLayer<Dtype>::CopyDiff(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) 
{
  // mask
  if(bottom.size() == 3) {
    const Dtype* mask = bottom[2]->cpu_data();
    const int batch_num = bottom[2]->num();
    const int sub_count = bottom[2]->count() / batch_num;
    CHECK_EQ(sub_count, this->key_point_num_);
    
    // Use mask to filter out some outliers
    Dtype* diff_data = this->diff_.mutable_cpu_data();
    for(int n = 0; n < batch_num; n++) {
      for(int sc = 0; sc < sub_count; sc++) {
        const int mask_offset = bottom[2]->offset(n, sc);
        const int diff_offset = this->diff_.offset(n, sc);
        caffe_scal(
            this->heat_num_, 
            mask[mask_offset],
            diff_data + diff_offset
        );
      }
    }
  }

  // Copy
  for (int idx = 0; idx < 2; ++idx) {
    if (propagate_down[idx]) {
      const Dtype sign = (idx == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[idx]->num();
      caffe_cpu_axpby(
          bottom[idx]->count(),                 // count
          alpha,                                // alpha
          this->diff_.cpu_data(),               // a
          Dtype(0),                             // beta
          bottom[idx]->mutable_cpu_diff()       // b
      );  
    }
  }
}

/**
 * @brief bottom[0] is predicted blob, bottom[1] is ground truth blob
 * But sometimes, bottom[1] may be another predicted blob (as in siamese network)
*/
template <typename Dtype>
void PoseHeatMapLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) 
{
  // Copy
  this->CopyDiff(top, propagate_down, bottom);
}

#ifdef CPU_ONLY
STUB_GPU(PoseHeatMapLossLayer);
#endif

INSTANTIATE_CLASS(PoseHeatMapLossLayer);
REGISTER_LAYER_CLASS(PoseHeatMapLoss);

}  // namespace caffe