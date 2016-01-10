 // Copyright 2015 DDK

#include <iostream>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <cfloat>
#include <vector>
#include <string>

#include "caffe/pose_estimation_layers.hpp"
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/common.hpp"
#include "caffe/global_variables.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
Dtype PoseHeatMapLossLayer<Dtype>::PrintLoss_gpu() 
{
  Dtype batch_loss;
  caffe_gpu_dot(
      this->diff_.count(), 
      this->diff_.gpu_data(), 
      this->diff_.gpu_data(), 
      &batch_loss
  );
  
  const Dtype num = Dtype(this->diff_.num() + 0.);
  const Dtype per_frame_loss = batch_loss / num;
  const Dtype key_point_num = Dtype(this->key_point_num_ + 0.);
  const Dtype per_heat_map_loss = per_frame_loss / key_point_num;

  LOG(INFO) << "layer name: " << this->layer_param_.name();
  LOG(INFO) << "iter: " << GlobalVars::caffe_iter();
  LOG(INFO) << "learn_lr: " << GlobalVars::learn_lr();
  LOG(INFO) << "euclidean loss (batch): " << batch_loss ;
  LOG(INFO) << "euclidean loss (frame): " << per_frame_loss;
  LOG(INFO) << "euclidean loss (joint): " << per_heat_map_loss;
  LOG(INFO) << "heat_map_loss_emphase_type: " << loss_emphase_type_;
  LOG(INFO) << "heat_score_thres: " << heat_score_thres_;
  LOG(INFO) << "prob_num: " << this->prob_num_;
  LOG(INFO);

  return batch_loss;
}

/* 
  in1: prediction
  in2: ground truth
  out: diff
*/

// <=0: default - consider ground truths and background
template <typename Dtype>
__global__ void PoseHeatMapLossForward0(const int n,
    const Dtype* in1, const Dtype* in2, Dtype* out) {
  CUDA_KERNEL_LOOP(idx, n) {
    out[idx] = in1[idx] - in2[idx];
  }
}

// 1: only consider the ground truths
template <typename Dtype>
__global__ void PoseHeatMapLossForward1(const int n,
    const Dtype* in1, const Dtype* in2, Dtype* out,
    const Dtype kErrorBound) {
  CUDA_KERNEL_LOOP(idx, n) {
    out[idx] = in1[idx] - in2[idx];
    if(in2[idx] < kErrorBound) {
      out[idx] = 0;
    }
  }
}

// 2: consider ground truths and background, and scale ground truths
template <typename Dtype>
__global__ void PoseHeatMapLossForward2(const int n,
    const Dtype* in1, const Dtype* in2, Dtype* out, 
    const Dtype eof, const Dtype kErrorBound) {
  CUDA_KERNEL_LOOP(idx, n) {
    out[idx] = in1[idx] - in2[idx];
    if(in2[idx] > kErrorBound) {
      out[idx] *= eof;
    }
  }
}

// 3|4: consider a littel bit of  backgrounds and ground truths(but scale them)
// offset: mark the offset of in(s) and out
template <typename Dtype>
__global__ void PoseHeatMapLossForward34(const int n,
    const Dtype* in1, const Dtype* in2, Dtype* out, 
    const unsigned int* rand_mask, const unsigned int uint_thres,
    const int offset, const Dtype fg_eof, const Dtype bg_eof,
    const int prob_num, const Dtype heat_score_thres, const Dtype kErrorBound)
{
  CUDA_KERNEL_LOOP(idx, n) {
    // compute diff
    const int o = offset + idx;
    out[o] = in1[o] - in2[o]; // pred - gt

    // Remove most of backgrounds because of ratio 
    // or only scale the ground truths
    if(in2[o] > kErrorBound) {
      out[o] *= fg_eof;  

    // the backgrounds keep the same (since bg_eof is always 1)
    } else if(rand_mask[idx] <= uint_thres) {
      out[o] *= bg_eof;  // bg_eof: always be 1

    // select the hard negative from predicted data
    } else if(in1[o] >= heat_score_thres) {
      if(rand_mask[idx] % prob_num) {
        out[o] = Dtype(0);
      }

    // neglect
    } else {
      out[o] = Dtype(0);
    }
  }
}

template <typename Dtype>
void PoseHeatMapLossLayer<Dtype>::CheckRandNum_gpu()
{
  Dtype zero_num = Dtype(0);
  caffe_copy(
      this->heat_num_, 
      this->rand_vec_.gpu_data(), 
      this->rand_vec_.mutable_cpu_data());

  for(int rv = 0; rv < this->heat_num_; rv++) {
    if(this->rand_vec_.cpu_data()[rv] <= this->uint_thres_) {
      LOG(INFO) << "random number(float): " 
          << this->rand_vec_.cpu_data()[rv]; 
      zero_num++; 
    }
  }

  LOG(INFO) << "ratio: " << this->ratio_;
  LOG(INFO) << "uint_thres: " << this->uint_thres_;
  LOG(INFO) << "zero_num: " << zero_num;
}

template <typename Dtype>
void PoseHeatMapLossLayer<Dtype>::ComputesHeatMapLoss_gpu(
      const vector<Blob<Dtype>*>& bottom) 
{
  const int num = bottom[0]->num();
  const int count = bottom[0]->count();
  const int heat_map_channels = bottom[0]->channels();
  CHECK_LE(this->key_point_num_, heat_map_channels);
  
  /* Get data&labels pointers */
  const Dtype* pred_data = bottom[0]->gpu_data();
  const Dtype* gt_data = bottom[1]->gpu_data();
  Dtype* diff_data = this->diff_.mutable_gpu_data();

  // const Dtype Zero = Dtype(0);
  const Dtype kErrorBound = Dtype(1e-4);  

  /* Default */
  if(this->loss_emphase_type_ <= 0) {
    PoseHeatMapLossForward0<Dtype><<<CAFFE_GET_BLOCKS(count), 
        CAFFE_CUDA_NUM_THREADS>>>(
          count, 
          pred_data,
          gt_data, 
          diff_data
    );
    CUDA_POST_KERNEL_CHECK;

  /* Only consider ground truths */
  } else if(this->loss_emphase_type_ == 1) {
    PoseHeatMapLossForward1<Dtype><<<CAFFE_GET_BLOCKS(count), 
        CAFFE_CUDA_NUM_THREADS>>>(
          count, 
          pred_data,
          gt_data, 
          diff_data,
          kErrorBound
    );
    CUDA_POST_KERNEL_CHECK;

  /* Consider backgrounds and ground truths(but scale them) */
  } else if(this->loss_emphase_type_ == 2) {
    PoseHeatMapLossForward2<Dtype><<<CAFFE_GET_BLOCKS(count), 
        CAFFE_CUDA_NUM_THREADS>>>(
          count, 
          pred_data,
          gt_data, 
          diff_data, 
          this->fg_eof_,
          kErrorBound
    );
    CUDA_POST_KERNEL_CHECK;

  /* Consider a littel bit of backgrounds 
   * and ground truths(but rescale them by fg_eof or bg_eof) 
   */
  } else if(this->loss_emphase_type_ == 3 || this->loss_emphase_type_ == 4) {
    /* get foreground eof */
    Dtype fg_eof2 = this->loss_emphase_type_ == 4 ? this->fg_eof_ : Dtype(1);
    // get ptr
    unsigned int* rand_mask = static_cast<unsigned int*>(
        this->rand_vec_.mutable_gpu_data());
    // get offset
    int hm_offset = 0;
    for(int np = 0; np < num; np++) {
      for(int kpn = 0; kpn < heat_map_channels; kpn++) {
        /* Create random numbers
         * Reference: src/caffe/layers/dropout_layer.cu 
         */
        caffe_gpu_rng_uniform(this->heat_num_, rand_mask);

        // Compute diff 
        PoseHeatMapLossForward34<Dtype>
          <<<CAFFE_GET_BLOCKS(this->heat_num_), 
            CAFFE_CUDA_NUM_THREADS>>>(
              this->heat_num_, 
              pred_data,
              gt_data, 
              diff_data,
              rand_mask, 
              this->uint_thres_, 
              hm_offset, 
              fg_eof2, 
              this->bg_eof_,
              this->prob_num_,
              Dtype(this->heat_score_thres_),
              kErrorBound
        );
        CUDA_POST_KERNEL_CHECK;

        // Add offset
        hm_offset += this->heat_num_;
      }
    }

    CHECK_EQ(hm_offset, count) << "does not match the size of each blob in bottom";
  } else {
    if(this->loss_emphase_type_) {
      LOG(INFO) << "loss_emphase_type_: " << this->loss_emphase_type_;
      NOT_IMPLEMENTED;
    }
  }
}

// bottom[0]: prediction
// bottom[1]: ground truths
template <typename Dtype>
void PoseHeatMapLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
  // Compute diff
  if(this->loss_emphase_type_ == 5) {
    this->ComputesHeatMapLoss(bottom);
  } else {
    this->ComputesHeatMapLoss_gpu(bottom);
  }
  
  // Print loss
  Dtype loss = this->PrintLoss_gpu();
  // Set loss
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void PoseHeatMapLossLayer<Dtype>::CopyDiff_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) 
{
  // mask
  if(bottom.size() == 3) {
    const Dtype* mask = bottom[2]->cpu_data();
    const int batch_num = bottom[2]->num();
    const int sub_count = bottom[2]->count() / batch_num;
    CHECK_EQ(sub_count, this->key_point_num_);
    // Use mask to filter out some outliers
    for(int n = 0; n < batch_num; n++) {
      for(int sc = 0; sc < sub_count; sc++) {
        const int mask_offset = bottom[2]->offset(n, sc);
        const int diff_offset = this->diff_.offset(n, sc);
        caffe_gpu_scal(
            this->heat_num_, 
            mask[mask_offset],
            this->diff_.mutable_gpu_data() + diff_offset
        );
      }
    }  
  }
  // Copy
  const Dtype alpha = top[0]->cpu_diff()[0] / bottom[0]->num();
  for (int idx = 0; idx < 2; ++idx) {
    if (propagate_down[idx]) {
      const Dtype sign = (idx == 0) ? 1 : -1;
      /* Copy */
      caffe_gpu_axpby(
          bottom[idx]->count(),                 // count
          sign * alpha,                         // alpha
          this->diff_.gpu_data(),               // a
          Dtype(0),                             // beta
          bottom[idx]->mutable_gpu_diff()       // b
      );  
    }
  }
}

template <typename Dtype>
void PoseHeatMapLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, 
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) 
{
  // Copy
  this->CopyDiff_gpu(top, propagate_down, bottom);
}

INSTANTIATE_LAYER_GPU_FUNCS(PoseHeatMapLossLayer);

}  // namespace caffe