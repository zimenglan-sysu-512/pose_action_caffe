#include <vector>
#include <algorithm>
#include <cfloat>

#include "caffe/layer.hpp"
#include "caffe/common.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/fast_rcnn_action_layers.hpp"

namespace caffe {

// bottom[0]: secondary regions' scores, like fc8, the pre-layer of softmax layer
// bottom[1]: n_rois_regions (the number of secondary regions of per image, produced by data layer)
template <typename Dtype>
void SecondaryRegionsOpScoresLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) 
{
  CHECK(this->layer_param_.has_secondary_regions_op_scores_param())
      << "must has secondary_regions_op_scores_param";
  // 
  const SecondaryRegionsOpScoresParameter& secondary_regions_op_scores_param = 
      this->layer_param_.secondary_regions_op_scores_param();

  if(bottom.size() == 1) {
    CHECK(secondary_regions_op_scores_param.has_n_secondary_regions())
        << "n_secondary_regions_ should be specified (the number of secondary regions of one image)";
    // 
    CHECK_GT(secondary_regions_op_scores_param.n_secondary_regions(), 0)
        << "n_secondary_regions_ should be specified (the number of secondary regions of one image)";
    // bottom[0]->num(): the total number of secondary regions per iteration
    CHECK_LE(secondary_regions_op_scores_param.n_secondary_regions(), bottom[0]->num())
        << "n_secondary_regions_ should be specified (the number of secondary regions of one image)";
  } else if(bottom.size() == 2) {
    const int n_secondary_regions = int(bottom[1]->cpu_data()[0]);
    LOG(INFO) << "n_secondary_regions: " << n_secondary_regions;

    if(n_secondary_regions) {
      CHECK_EQ(n_secondary_regions, bottom[0]->num());
    }
    CHECK_EQ(bottom[1]->channels(),  1);
    CHECK_EQ(bottom[1]->height(),  1);
    CHECK_EQ(bottom[1]->width(),  1);

    LOG(INFO) << "n_secondary_regions2: " << n_secondary_regions;

    // bottom[1]->num(): equal to ims_per_batch
    for(int idx = 0; idx < bottom[1]->count(); idx++) {
      CHECK_EQ(n_secondary_regions, bottom[1]->cpu_data()[idx]);
    }
    LOG(INFO) << "n_secondary_regions3: " << n_secondary_regions;
  } else {
    LOG(FATAL) << "wrong prototxt settings";
  }
}

template <typename Dtype>
void SecondaryRegionsOpScoresLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) 
{
  const int num = bottom[0]->num();
  const SecondaryRegionsOpScoresParameter& secondary_regions_op_scores_param = 
      this->layer_param_.secondary_regions_op_scores_param();
  if(bottom.size() == 1) {
    this->n_secondary_regions_ = secondary_regions_op_scores_param.n_secondary_regions();
  } else {
    this->n_secondary_regions_ = int(bottom[1]->cpu_data()[0]);
    // 为了防止空data的时候初始化网络
    if(!this->n_secondary_regions_) {
      this->n_secondary_regions_ = bottom[1]->num();
      this->init_first_ = false;
    }
  }
  
  if(secondary_regions_op_scores_param.has_has_scene_branch()) {
    if(secondary_regions_op_scores_param.has_scene_branch()) {
      this->n_secondary_regions_ -= 1;
    }
  }
  // 为了防止空data的时候初始化网络
  if(!this->n_secondary_regions_ && !this->init_first_) {
    this->n_secondary_regions_ = bottom[1]->num();
    this->init_first_ = true;
  }

  CHECK_LE(this->n_secondary_regions_, num)
      << "n_secondary_regions_ should be less than or equal to `num`";
  CHECK(!(num % this->n_secondary_regions_))
      << "num: " << num 
      << ", n_secondary_regions_: " << this->n_secondary_regions_
      << ", -- `num == k * n_secondary_regions_` is true, where k = 1, 2, ...";

  this->ims_per_batch_ = num / this->n_secondary_regions_;
  const int channels = bottom[0]->channels();
  const int height = bottom[0]->height();
  const int width = bottom[0]->width();
  const int top_num = this->ims_per_batch_; //
  top[0]->Reshape(top_num, channels, height, width);

  this->has_max_op_ = false;
  if(this->layer_param_.secondary_regions_op_scores_param().op_scores() ==
      SecondaryRegionsOpScoresParameter_OpScoresMethod_MAX) {
    this->has_max_op_ = true;
  }
  
  if(!this->has_max_op_) {
    return;
  }
  // top[1]: max_selected_secondary_regions_inds
  // top[2]: inds of max pooling
  if(top.size() > 1) {
    top[1]->ReshapeLike(*top[0]);
  }
  if(top.size() == 3) {
    top[2]->ReshapeLike(*top[0]);
  } else {
    max_idx_.Reshape(top_num, channels, height, width);
  }

  if(top.size() > 3) {
    LOG(INFO) << "invalid operation and top blobs";
    NOT_IMPLEMENTED;
  }
}

// top[0]: secondary_max_score
// top[1]: max_selected_secondary_regions_inds, if have
//    record the index of selected secondary region, for regions extractions
//    as input to `FusionRegionsLayer'
// top[2]: secondary_max_score_inds, if have
//    record the pos of selected secondary regions in the blob, for back-propogation
template <typename Dtype>
void SecondaryRegionsOpScoresLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) 
{
  // 
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  Dtype* max_selected_secondary_regions_inds = NULL;
  if(top.size() > 1) {
    max_selected_secondary_regions_inds = top[1]->mutable_cpu_data();
  }

  // 
  const Dtype Zero = Dtype(0);
  const Dtype ScalFactor = Dtype(1) / Dtype(this->n_secondary_regions_ + 0.);
  // 
  const int top_count = top[0]->count();
  const int sub_count = top_count / top[0]->num();
  CHECK_EQ(sub_count, bottom[0]->count() / bottom[0]->num());

  // We'll output the mask to top[1] if it's of size >1.
  const bool has_use_top_mask = top.size() > 2;
  int* mask = NULL;  // suppress warnings about uninitalized variables
  Dtype* top_mask = NULL;

  // Start
  switch (this->layer_param_.secondary_regions_op_scores_param().op_scores()) {
    // 
    case SecondaryRegionsOpScoresParameter_OpScoresMethod_MAX:
      // Initialize -- mask
      if (has_use_top_mask) {
        top_mask = top[2]->mutable_cpu_data();
        caffe_set(top_count, Dtype(-1), top_mask);
      } else {
        mask = max_idx_.mutable_cpu_data();
        caffe_set(top_count, -1, mask);
      }
      // Initialize -- top_data
      caffe_set(top_count, Dtype(-FLT_MAX), top_data);

      // The main loop
      for(int ipb = 0; ipb < this->ims_per_batch_; ipb++) {
        // add offset
        int ipb_offset = top[0]->offset(ipb);
        top_data = top[0]->mutable_cpu_data() + ipb_offset;
        if(top.size() > 1) {
          max_selected_secondary_regions_inds = 
              top[1]->mutable_cpu_data() + ipb_offset;
        }

        // add offset
        if (has_use_top_mask) {
          top_mask = top[2]->mutable_cpu_data() + ipb_offset;
        } else {
          mask = max_idx_.mutable_cpu_data() + ipb_offset;
        }

        int nsr_num = ipb * this->n_secondary_regions_;
        for(int nsr = 0; nsr < this->n_secondary_regions_; nsr++) {
          // add offset
          int nsr_offset = bottom[0]->offset(nsr_num + nsr);
          bottom_data = bottom[0]->cpu_data() + nsr_offset;

          for(int sct = 0; sct < sub_count; sct++) {
            if(bottom_data[sct] > top_data[sct]) {
              // set value
              top_data[sct] = bottom_data[sct];
              if(top.size() > 1) {
                max_selected_secondary_regions_inds[sct] = nsr_num + nsr;
              }

              if (has_use_top_mask) {
                top_mask[sct] = static_cast<Dtype>(nsr_offset + sct);
              } else {
                mask[sct] = nsr_offset + sct;
              }
            }
          }
        }
      }

      break;

    // 
    case SecondaryRegionsOpScoresParameter_OpScoresMethod_SUM:
      // Initialize
      caffe_set(top_count, Zero, top_data);

      // The main loop
      for(int ipb = 0; ipb < this->ims_per_batch_; ipb++) {
        // add offset
        int ipb_offset = top[0]->offset(ipb);
        top_data = top[0]->mutable_cpu_data() + ipb_offset;
        
        int nsr_num = ipb * this->n_secondary_regions_;
        for(int nsr = 0; nsr < this->n_secondary_regions_; nsr++) {
          // add offset
          int nsr_offset = bottom[0]->offset(nsr_num + nsr);
          bottom_data = bottom[0]->cpu_data() + nsr_offset;

          for(int sct = 0; sct < sub_count; sct++) {
            // add value -- accumulate
            top_data[sct] += bottom_data[sct];
          } // end sct

        } // end nsr
      } // end ipb
     
      break;
    
    // 
    case SecondaryRegionsOpScoresParameter_OpScoresMethod_AVE:
      // Initialize
      caffe_set(top_count, Zero, top_data);

      // The main loop
      for(int ipb = 0; ipb < this->ims_per_batch_; ipb++) {
        // add offset
        int ipb_offset = top[0]->offset(ipb);
        top_data = top[0]->mutable_cpu_data() + ipb_offset;
        
        int nsr_num = ipb * this->n_secondary_regions_;
        for(int nsr = 0; nsr < this->n_secondary_regions_; nsr++) {
          // add offset
          int nsr_offset = bottom[0]->offset(nsr_num + nsr);
          bottom_data = bottom[0]->cpu_data() + nsr_offset;

          for(int sct = 0; sct < sub_count; sct++) {
            // add value -- accumulate
            top_data[sct] += bottom_data[sct];
          } // end sct
        } // end nsr
      } // end ipb
      
      // ave
      caffe_scal(
          top_count, 
          ScalFactor, 
          top[0]->mutable_cpu_data()
      );

      break;

    // 
    default:
      LOG(FATAL) << "Unknown op_scores method.";
  }
}



template <typename Dtype>
void SecondaryRegionsOpScoresLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) 
{
  if(!propagate_down[0]) {
    return;
  }

  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

  // Different pooling methods. We explicitly do the switch outside the for
  // loop to save time, although this results in more codes.
  const Dtype Zero = Dtype(0);
  const int nOne = -1;
  const Dtype ScalFactor = Dtype(1) / Dtype(this->n_secondary_regions_ + 0.);
  // 
  const int top_count = top[0]->count();
  const int sub_count = top_count / top[0]->num();
  CHECK_EQ(sub_count, bottom[0]->count() / bottom[0]->num());

  // Initialize
  caffe_set(bottom[0]->count(), Zero, bottom_diff);

  // We'll output the mask to top[1] if it's of size > 2.
  const bool has_use_top_mask = top.size() > 2;
  const int* mask = NULL;  // suppress warnings about uninitialized variables
  const Dtype* top_mask = NULL;
  

  // Main Loop
  switch (this->layer_param_.secondary_regions_op_scores_param().op_scores()) {
    // MAX
    case SecondaryRegionsOpScoresParameter_OpScoresMethod_MAX:
      // The main loop
      if (has_use_top_mask) {
        top_mask = top[2]->cpu_data();
      } else {
        mask = max_idx_.cpu_data();
      }

      // The main loop
      for(int ipb = 0; ipb < this->ims_per_batch_; ipb++) {
        // add offset
        int ipb_offset = top[0]->offset(ipb);
        top_diff = top[0]->cpu_diff() + ipb_offset;

        // add offset
        if (has_use_top_mask) {
          top_mask = top[2]->cpu_data() + ipb_offset;
        } else {
          mask = max_idx_.cpu_data() + ipb_offset;
        }

        for(int sct = 0; sct < sub_count; sct++) {
          // get offset
          const int bottom_index = has_use_top_mask ? top_mask[sct] : mask[sct];

          CHECK_GT(bottom_index, nOne);
          // set diff value -- accumulated
          bottom_diff[bottom_index] += top_diff[sct];
        }
      }

      break;

    // SUM
    case SecondaryRegionsOpScoresParameter_OpScoresMethod_SUM:
      // 
      for(int ipb = 0; ipb < this->ims_per_batch_; ipb++) {
        // add offset
        int ipb_offset = top[0]->offset(ipb);
        top_diff = top[0]->cpu_diff() + ipb_offset;
        
        int nsr_num = ipb * this->n_secondary_regions_;
        for(int nsr = 0; nsr < this->n_secondary_regions_; nsr++) {
          // add offset
          int nsr_offset = bottom[0]->offset(nsr_num + nsr);
          bottom_diff = bottom[0]->mutable_cpu_diff() + nsr_offset;

          for(int sct = 0; sct < sub_count; sct++) {
            // add diff value -- accumulated
            // bottom_diff[sct] += (top_diff[sct] * ScalFactor);
            bottom_diff[sct] += top_diff[sct];
          } // end sct
        } // end nsr
      } // end ipb
      // ave
      caffe_scal(
          bottom[0]->count(),
          ScalFactor, 
          bottom[0]->mutable_cpu_diff()
      );

      break;

    // AVE
    case SecondaryRegionsOpScoresParameter_OpScoresMethod_AVE:
      // 
      for(int ipb = 0; ipb < this->ims_per_batch_; ipb++) {
        // add offset
        int ipb_offset = top[0]->offset(ipb);
        top_diff = top[0]->cpu_diff() + ipb_offset;
        
        int nsr_num = ipb * this->n_secondary_regions_;
        for(int nsr = 0; nsr < this->n_secondary_regions_; nsr++) {
          // add offset
          int nsr_offset = bottom[0]->offset(nsr_num + nsr);
          bottom_diff = bottom[0]->mutable_cpu_diff() + nsr_offset;

          for(int sct = 0; sct < sub_count; sct++) {
            // add diff value -- accumulated
            // bottom_diff[sct] += (top_diff[sct] * ScalFactor);
            bottom_diff[sct] += top_diff[sct];
          } // end sct
        } // end nsr
      } // end ipb
      // ave
      caffe_scal(
          bottom[0]->count(),
          ScalFactor, 
          bottom[0]->mutable_cpu_diff()
      );

      break;

    // default
    default:
      LOG(FATAL) << "Unknown op_scores method.";
  }

}


#ifdef CPU_ONLY
STUB_GPU(SecondaryRegionsOpScoresLayer);
#endif

INSTANTIATE_CLASS(SecondaryRegionsOpScoresLayer);
REGISTER_LAYER_CLASS(SecondaryRegionsOpScores);


}  // namespace caffe