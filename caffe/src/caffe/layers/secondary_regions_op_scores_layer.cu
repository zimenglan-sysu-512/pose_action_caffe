#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/common.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"

#include "caffe/vision_layers.hpp"
#include "caffe/fast_rcnn_action_layers.hpp"


namespace caffe {



template <typename Dtype>
__global__ void MaxPoolForward(
    const int nthreads/*equal to sub_count*/, 
    const Dtype* bottom_data,
    Dtype* top_data,
    int* mask, 
    Dtype* top_mask,
    const bool has_use_top_mask,
    const int sub_count,
    const int nsr_offset) 
{
  CUDA_KERNEL_LOOP(sct, nthreads) {
    if(bottom_data[sct] > top_data[sct]) {
      // set value
      top_data[sct] = bottom_data[sct];

      if (has_use_top_mask) {
        top_mask[sct] = static_cast<Dtype>(nsr_offset + sct);
      } else {
        mask[sct] = nsr_offset + sct;
      }
    }
  }
}


template <typename Dtype>
__global__ void MaxPoolForward2(
    const int nthreads/*equal to sub_count*/, 
    const Dtype* bottom_data,
    Dtype* top_data,
    Dtype* max_selected_secondary_regions_inds, 
    int* mask, 
    Dtype* top_mask,
    const bool has_use_top_mask,
    const int sub_count,
    const int nsr_offset,
    const int mssr_ind) 
{
  CUDA_KERNEL_LOOP(sct, nthreads) {
    if(bottom_data[sct] > top_data[sct]) {
      // set value
      top_data[sct] = bottom_data[sct];
      max_selected_secondary_regions_inds[sct] = mssr_ind;

      if (has_use_top_mask) {
        top_mask[sct] = static_cast<Dtype>(nsr_offset + sct);
      } else {
        mask[sct] = nsr_offset + sct;
      }
    }
  }
}

template <typename Dtype>
void SecondaryRegionsOpScoresLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) 
{
  // Get data ptr
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  
  // 
  const Dtype Zero = Dtype(0);
  const Dtype alpha = Dtype(1.);
  const Dtype beta = Dtype(1.);
  const Dtype ScalFactor = Dtype(1) / Dtype(this->n_secondary_regions_ + 0.);
  // 
  const int top_count = top[0]->count();
  const int sub_count = top_count / top[0]->num();
  CHECK_EQ(sub_count, bottom[0]->count() / bottom[0]->num());

  // We'll output the mask to top[2] if it's of size > 2.
  const bool has_use_top_mask = top.size() > 2;
  const bool has_secondary_regions_inds = top.size() > 1;
  int* mask = NULL;       // use cpu_data to initialize
  Dtype* top_mask = NULL;
  Dtype* max_selected_secondary_regions_inds = NULL;
  if(has_secondary_regions_inds) {
    max_selected_secondary_regions_inds = top[1]->mutable_cpu_data();
  }


  // Main Loop
  switch (this->layer_param_.secondary_regions_op_scores_param().op_scores()) {
    // Max
    case SecondaryRegionsOpScoresParameter_OpScoresMethod_MAX:
      if (has_use_top_mask) {
        top_mask = top[2]->mutable_gpu_data();
        caffe_gpu_set(top_count, Dtype(-1), top_mask);
      } else {
        mask = max_idx_.mutable_cpu_data();
        const int nOne = -1;
        caffe_set(top_count, nOne, mask);
        caffe_copy(
            top_count, 
            max_idx_.cpu_data(), 
            max_idx_.mutable_gpu_data()
        );
        mask = max_idx_.mutable_gpu_data();
      }
      // Initialize
      caffe_gpu_set(top_count, Dtype(-FLT_MAX), top_data);

      for(int ipb = 0; ipb < this->ims_per_batch_; ipb++) {
        // add offset
        int ipb_offset = top[0]->offset(ipb);
        top_data = top[0]->mutable_gpu_data() + ipb_offset;
        if(has_secondary_regions_inds) {
          max_selected_secondary_regions_inds = 
              top[1]->mutable_gpu_data() + ipb_offset;
        }

        // add offset
        if (has_use_top_mask) {
          top_mask = top[2]->mutable_gpu_data() + ipb_offset;
        } else {
          mask =max_idx_.mutable_gpu_data() + ipb_offset;
        }

        int nsr_num = ipb * this->n_secondary_regions_;
        for(int nsr = 0; nsr < this->n_secondary_regions_; nsr++) {
          // add offset
          int nsr_offset = bottom[0]->offset(nsr_num + nsr);
          bottom_data = bottom[0]->gpu_data() + nsr_offset;

          if(!has_secondary_regions_inds) {
            // NOLINT_NEXT_LINE(whitespace/operators)
            MaxPoolForward<Dtype><<<CAFFE_GET_BLOCKS(sub_count), 
              CAFFE_CUDA_NUM_THREADS>>>(
                sub_count, 
                bottom_data, 
                top_data,
                mask, 
                top_mask,
                has_use_top_mask,
                sub_count,
                nsr_offset
              );
          } else {
            // NOLINT_NEXT_LINE(whitespace/operators)
            MaxPoolForward2<Dtype><<<CAFFE_GET_BLOCKS(sub_count), 
              CAFFE_CUDA_NUM_THREADS>>>(
                sub_count, 
                bottom_data, 
                top_data,
                max_selected_secondary_regions_inds,
                mask, 
                top_mask,
                has_use_top_mask,
                sub_count,
                nsr_offset,
                nsr_num + nsr
              );
          }
        }
      }

      break;
    
    // Sum
    case SecondaryRegionsOpScoresParameter_OpScoresMethod_SUM:
      // Initialize to zero
      caffe_gpu_set(top_count, Zero, top_data);

      for(int ipb = 0; ipb < this->ims_per_batch_; ipb++) {
        // add offset
        int ipb_offset = top[0]->offset(ipb);
        top_data = top[0]->mutable_gpu_data() + ipb_offset;

        int nsr_num = ipb * this->n_secondary_regions_;
        for(int nsr = 0; nsr < this->n_secondary_regions_; nsr++) {
          // add offset
          int nsr_offset = bottom[0]->offset(nsr_num + nsr);
          bottom_data = bottom[0]->gpu_data() + nsr_offset;

          // 
          caffe_gpu_axpby(
              sub_count,
              alpha,
              bottom_data,
              beta,
              top_data
          );
        }
      }
    
      break;

    // Ave
    case SecondaryRegionsOpScoresParameter_OpScoresMethod_AVE:
      // Initialize to zero
      caffe_gpu_set(top_count, Zero, top_data);

      for(int ipb = 0; ipb < this->ims_per_batch_; ipb++) {
        // add offset
        int ipb_offset = top[0]->offset(ipb);
        top_data = top[0]->mutable_gpu_data() + ipb_offset;

        int nsr_num = ipb * this->n_secondary_regions_;
        for(int nsr = 0; nsr < this->n_secondary_regions_; nsr++) {
          // add offset
          int nsr_offset = bottom[0]->offset(nsr_num + nsr);
          bottom_data = bottom[0]->gpu_data() + nsr_offset;
          // 
          caffe_gpu_axpby(
              sub_count,
              alpha,
              bottom_data,
              beta,
              top_data
          );
        }
      }

      // 
      caffe_gpu_scal(
          top_count, 
          ScalFactor,
          top[0]->mutable_gpu_data()
      );

      break;

    // default
    default:
      LOG(FATAL) << "Unknown op_scores method.";
  }
  // 
  CUDA_POST_KERNEL_CHECK;
}




/* ********************************************************************** */




template <typename Dtype>
__global__ void MaxPoolBackward_Top_Mask(
    const int nthreads, 
    const Dtype* top_diff,
    const Dtype* top_mask, 
    Dtype* bottom_diff) 
{ CUDA_KERNEL_LOOP(nts, nthreads) {
    // get idx
    const int idx = top_mask[nts];

    // set diff value -- accumulated
    bottom_diff[idx] += top_diff[nts];
  }
}


template <typename Dtype>
__global__ void MaxPoolBackward_Mask(
    const int nthreads, 
    const Dtype* top_diff,
    const int* mask, 
    Dtype* bottom_diff) 
{
  CUDA_KERNEL_LOOP(nts, nthreads) {
    // get idx
    const int idx = mask[nts];

    // set diff value -- accumulated
    bottom_diff[idx] += top_diff[nts];
  }
}



template <typename Dtype>
void SecondaryRegionsOpScoresLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) 
{
  if(!propagate_down[0]) {
    return;
  }

  // 
  const Dtype* top_diff = top[0]->gpu_diff();

  // Initialize to zero
  const int count = bottom[0]->count();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();

  const Dtype alpha = Dtype(1.);
  const Dtype beta = Dtype(1.);
  const Dtype Zero = Dtype(0);
  const Dtype ScalFactor = Dtype(1) / Dtype(this->n_secondary_regions_ + 0.);
  // 
  const int top_count = top[0]->count();
  const int sub_count = top_count / top[0]->num();
  CHECK_EQ(sub_count, bottom[0]->count() / bottom[0]->num());

  // We'll output the mask to top[2] if it's of size > 2.
  const bool has_use_top_mask = top.size() > 2;
  const int* mask = NULL;
  const Dtype* top_mask = NULL;
  
  // 
  caffe_gpu_set(count, Zero, bottom_diff);

  // Main Loop
  switch (this->layer_param_.secondary_regions_op_scores_param().op_scores()) {
    // Max
    case SecondaryRegionsOpScoresParameter_OpScoresMethod_MAX:
      if (has_use_top_mask) {
        top_mask = top[2]->gpu_data();
        // 
        MaxPoolBackward_Top_Mask<Dtype><<<CAFFE_GET_BLOCKS(top_count), 
          CAFFE_CUDA_NUM_THREADS>>>(
            top_count, 
            top_diff, 
            top_mask, 
            bottom_diff
          );
      } else {
        mask = max_idx_.gpu_data();
        // 
        MaxPoolBackward_Mask<Dtype><<<CAFFE_GET_BLOCKS(top_count), 
          CAFFE_CUDA_NUM_THREADS>>>(
            top_count, 
            top_diff, 
            mask, 
            bottom_diff
          );
      }

      break;
    
    // Sum
    case SecondaryRegionsOpScoresParameter_OpScoresMethod_SUM:
      // 
      for(int ipb = 0; ipb < this->ims_per_batch_; ipb++) {
        // add offset
        int ipb_offset = top[0]->offset(ipb);
        top_diff = top[0]->gpu_diff() + ipb_offset;
        
        int nsr_num = ipb * this->n_secondary_regions_;
        for(int nsr = 0; nsr < this->n_secondary_regions_; nsr++) {
          // add offset
          int nsr_offset = bottom[0]->offset(nsr_num + nsr);
          bottom_diff = bottom[0]->mutable_gpu_diff() + nsr_offset;

          // 
          caffe_gpu_axpby(
              sub_count,
              alpha,
              top_diff,
              beta,
              bottom_diff
          );
        } // end nsr
      } // end ipb

      // 
      caffe_gpu_scal(
          count, 
          ScalFactor,
          bottom[0]->mutable_gpu_diff()
      );

      break;

    // Ave
    case SecondaryRegionsOpScoresParameter_OpScoresMethod_AVE:
      // 
      for(int ipb = 0; ipb < this->ims_per_batch_; ipb++) {
        // add offset
        int ipb_offset = top[0]->offset(ipb);
        top_diff = top[0]->gpu_diff() + ipb_offset;
        
        int nsr_num = ipb * this->n_secondary_regions_;
        for(int nsr = 0; nsr < this->n_secondary_regions_; nsr++) {
          // add offset
          int nsr_offset = bottom[0]->offset(nsr_num + nsr);
          bottom_diff = bottom[0]->mutable_gpu_diff() + nsr_offset;
          
          // 
          caffe_gpu_axpby(
              sub_count,
              alpha,
              top_diff,
              beta,
              bottom_diff
          );
        } // end nsr
      } // end ipb

      // 
      caffe_gpu_scal(
          count, 
          ScalFactor,
          bottom[0]->mutable_gpu_diff()
      );

      break;

    // default
    default:
      LOG(FATAL) << "Unknown op_scores method.";
  }

  CUDA_POST_KERNEL_CHECK;
}



INSTANTIATE_LAYER_GPU_FUNCS(SecondaryRegionsOpScoresLayer);



}  // namespace caffe