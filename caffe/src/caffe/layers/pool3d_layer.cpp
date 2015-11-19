#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/fast_rcnn_action_layers.hpp"

namespace caffe {

using std::min;
using std::max;

template <typename Dtype>
void Pooling3DLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) 
{
  const GroupPoolingParameter group_pooling_param = 
  		this->layer_param_.group_pooling_param();

  // kernel_size V.S. kernel_h/kernel_w
  if (group_pooling_param.global_pooling()) {
    CHECK(!(group_pooling_param.has_kernel_size() ||
      group_pooling_param.has_kernel_h() || group_pooling_param.has_kernel_w()))
      << "With Global_pooling: true Filter size cannot specified";
  } else {
    CHECK(!group_pooling_param.has_kernel_size() !=
      !(group_pooling_param.has_kernel_h() && group_pooling_param.has_kernel_w()))
      << "Filter size is kernel_size OR kernel_h and kernel_w; not both";
    CHECK(group_pooling_param.has_kernel_size() ||
      (group_pooling_param.has_kernel_h() && group_pooling_param.has_kernel_w()))
      << "For non-square filters both kernel_h and kernel_w are required.";
  }

  // padding
  CHECK((!group_pooling_param.has_pad() && group_pooling_param.has_pad_h()
      && group_pooling_param.has_pad_w())
      || (!group_pooling_param.has_pad_h() && !group_pooling_param.has_pad_w()))
      << "pad is pad OR pad_h and pad_w are required.";

  // stride
  CHECK((!group_pooling_param.has_stride() && group_pooling_param.has_stride_h()
      && group_pooling_param.has_stride_w())
      || (!group_pooling_param.has_stride_h() && !group_pooling_param.has_stride_w()))
      << "Stride is stride OR stride_h and stride_w are required.";


  // ###############################################################


  global_pooling_ = group_pooling_param.global_pooling();
  if (global_pooling_) {
    kernel_h_ = bottom[0]->height();
    kernel_w_ = bottom[0]->width();
  } else {
    if (group_pooling_param.has_kernel_size()) {
      kernel_h_ = kernel_w_ = group_pooling_param.kernel_size();
    } else {
      kernel_h_ = group_pooling_param.kernel_h();
      kernel_w_ = group_pooling_param.kernel_w();
    }
  }
  CHECK_GT(kernel_h_, 0) << "Filter dimensions cannot be zero.";
  CHECK_GT(kernel_w_, 0) << "Filter dimensions cannot be zero.";

  // padding
  if (!group_pooling_param.has_pad_h()) {
    pad_h_ = pad_w_ = group_pooling_param.pad();
  } else {
    pad_h_ = group_pooling_param.pad_h();
    pad_w_ = group_pooling_param.pad_w();
  }

  // stride
  if (!group_pooling_param.has_stride_h()) {
    stride_h_ = stride_w_ = group_pooling_param.stride();
  } else {
    stride_h_ = group_pooling_param.stride_h();
    stride_w_ = group_pooling_param.stride_w();
  }

  if (global_pooling_) {
    CHECK(pad_h_ == 0 && pad_w_ == 0 && stride_h_ == 1 && stride_w_ == 1)
      << "With Global_pooling: true; only pad = 0 and stride = 1";
  }

  // 
  if (pad_h_ != 0 || pad_w_ != 0) {
    CHECK(this->layer_param_.pooling_param().pool()
        == GroupPoolingParameter_PoolMethod_AVE
        || this->layer_param_.pooling_param().pool()
        == GroupPoolingParameter_PoolMethod_MAX)
        << "Padding implemented only for average and max pooling.";
    CHECK_LT(pad_h_, kernel_h_);
    CHECK_LT(pad_w_, kernel_w_);
  }

 	// 
  this->temporal_length_ = group_pooling_param.temporal_length();
  this->temporal_stride_ = group_pooling_param.temporal_stride();
}



template <typename Dtype>
void Pooling3DLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) 
{
  CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
      << "corresponding to (num, channels, height, width)";

  num_ = bottom[0]->num();
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();

  if (global_pooling_) {
    kernel_h_ = bottom[0]->height();
    kernel_w_ = bottom[0]->width();
  }

  // 
  pooled_height_ = static_cast<int>(ceil(static_cast<float>(
      height_ + 2 * pad_h_ - kernel_h_) / stride_h_)) + 1;
  pooled_width_ = static_cast<int>(ceil(static_cast<float>(
      width_ + 2 * pad_w_ - kernel_w_) / stride_w_)) + 1;
  pooled_channels_ = static_cast<int>(ceil(static_cast<float>(
  		channels_ - temporal_length_) / temporal_stride_)) + 1;
  CHECK_LT((pooled_channels_ - 1) * temporal_stride_, channels_);

  // 
  if (pad_h_ || pad_w_) {
    // If we have padding, ensure that the last pooling starts strictly
    // inside the image (instead of at the padding); otherwise clip the last.
    if ((pooled_height_ - 1) * stride_h_ >= height_ + pad_h_) {
      --pooled_height_;
    }
    if ((pooled_width_ - 1) * stride_w_ >= width_ + pad_w_) {
      --pooled_width_;
    }
    CHECK_LT((pooled_height_ - 1) * stride_h_, height_ + pad_h_);
    CHECK_LT((pooled_width_ - 1) * stride_w_, width_ + pad_w_);
  }

  // 
  top[0]->Reshape(num_, pooled_channels_, pooled_height_, pooled_width_);
  if (top.size() > 1) {
    top[1]->ReshapeLike(*top[0]);
  }

  // If max pooling, we will initialize the vector index part.
  if (this->layer_param_.pooling_param().pool() ==
      GroupPoolingParameter_PoolMethod_MAX && top.size() == 1) {
    max_idx_.Reshape(num_, pooled_channels_, pooled_height_, pooled_width_);
  }
  // If stochastic pooling, we will initialize the random index part.
  if (this->layer_param_.pooling_param().pool() ==
      GroupPoolingParameter_PoolMethod_STOCHASTIC) {
    rand_idx_.Reshape(num_, pooled_channels_, pooled_height_, pooled_width_);
  }
}




// TODO(Yangqing): Is there a faster way to do pooling in the channel-first
// case?
template <typename Dtype>
void Pooling3DLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) 
{
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int top_count = top[0]->count();

  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  // suppress warnings about uninitalized variables
  int* mask = NULL;  
  Dtype* top_mask = NULL;



  // Different pooling methods. We explicitly do the switch outside the for
  // loop to save time, although this results in more code.
  switch (this->layer_param_.pooling_param().pool()) {
  // Max
  case GroupPoolingParameter_PoolMethod_MAX:
    // Initialize the indices
    if (use_top_mask) {
      top_mask = top[1]->mutable_cpu_data();
      caffe_set(top_count, Dtype(-1), top_mask);
    } else {
      mask = max_idx_.mutable_cpu_data();
      caffe_set(top_count, -1, mask);
    }

    // Initialize
    caffe_set(top_count, Dtype(-FLT_MAX), top_data);

    // The main loop
    for (int n = 0; n < num_; ++n) {
    	// 
      for (int c = 0; c < pooled_channels_; ++c) {
        for (int ph = 0; ph < pooled_height_; ++ph) {
          for (int pw = 0; pw < pooled_width_; ++pw) {

            int hstart = ph * stride_h_ - pad_h_;
            int wstart = pw * stride_w_ - pad_w_;
            int cstart = c  * temporal_stride_;

            int hend = min(hstart + kernel_h_, height_);
            int wend = min(wstart + kernel_w_, width_);
            int cend = min(cstart + temporal_length_, channels_);

            hstart = max(hstart, 0);
            wstart = max(wstart, 0);
            cstart = max(cstart, 0);

            // ((n * channels() + c) * height() + h) * width() + w;
            // const int pool_index = ph * pooled_width_ + pw;
            const int pool_index = (c * pooled_height_ + ph) * pooled_width_ + pw;

            for(int c2 = cstart; c2 < cend; ++c2) {
	            for (int h = hstart; h < hend; ++h) {
	              for (int w = wstart; w < wend; ++w) {
	              	// 
	                // const int index = h * width_ + w;
	                const int index = (c2 * height_ + h) * width_ + w;

	                if (bottom_data[index] > top_data[pool_index]) {
	                  top_data[pool_index] = bottom_data[index];
	                  if (use_top_mask) {
	                    top_mask[pool_index] = static_cast<Dtype>(index);
	                  } else {
	                    mask[pool_index] = index;
	                  }
	                }
	              }
	            }
	          }	// end one cube

          }	// end pooled_width_
        }	// end pooled_height_
      }	// end all cubes -- pooled_channels_

      // compute offset
      bottom_data += bottom[0]->offset(1);
      top_data += top[0]->offset(1);
      if (use_top_mask) {
        top_mask += top[0]->offset(1);
      } else {
        mask += top[0]->offset(1);
      }
    }	// end batch -- num_
    break;

  // Average
  case GroupPoolingParameter_PoolMethod_AVE:
    for (int i = 0; i < top_count; ++i) {
      top_data[i] = 0;
    }
    // The main loop
    for (int n = 0; n < bottom[0]->num(); ++n) {
    	// 
      for (int c = 0; c < pooled_channels_; ++c) {
        for (int ph = 0; ph < pooled_height_; ++ph) {
          for (int pw = 0; pw < pooled_width_; ++pw) {

          	int hstart = ph * stride_h_ - pad_h_;
            int wstart = pw * stride_w_ - pad_w_;
            int cstart = c  * temporal_stride_;

            int hend = min(hstart + kernel_h_, height_ + pad_h_);
            int wend = min(wstart + kernel_w_, width_ + pad_w_);
            int cend = min(cstart + temporal_length_, channels_);

            hstart = max(hstart, 0);
            wstart = max(wstart, 0);
            cstart = max(cstart, 0);

            hend = min(hend, height_);
            wend = min(wend, width_);

            // ((n * channels() + c) * height() + h) * width() + w;
            // const int pool_index = ph * pooled_width_ + pw;
            const int pool_index = (c * pooled_height_ + ph) * pooled_width_ + pw;

           	for(int c2 = cstart; c2 < cend; ++c2) {
	            for (int h = hstart; h < hend; ++h) {
	              for (int w = wstart; w < wend; ++w) {
	              	// const int index = h * width_ + w;
	                const int index = (c2 * height_ + h) * width_ + w;
	                top_data[pool_index] += bottom_data[index];
	              }
	            }
	          }
	          // 
            int pool_size = (cend - cstart) * (hend - hstart) * (wend - wstart);
            top_data[pool_index] /= pool_size;

          }	// end pooled_width_
        }	// end pooled_height_
      } // end pooled_channels_

      // compute offset
      bottom_data += bottom[0]->offset(1);
      top_data += top[0]->offset(1);
    }	// end num_
    break;

  // Stochastic
  case GroupPoolingParameter_PoolMethod_STOCHASTIC:
    NOT_IMPLEMENTED;
    break;

  // default
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
}




template <typename Dtype>
void Pooling3DLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }

  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  
  // Different pooling methods. We explicitly do the switch outside the for
  // loop to save time, although this results in more codes.
  caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);
  
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  const int* mask = NULL;  // suppress warnings about uninitialized variables
  const Dtype* top_mask = NULL;
  
  // 
  switch (this->layer_param_.pooling_param().pool()) {
  // Max
  case GroupPoolingParameter_PoolMethod_MAX:
    // The main loop
    if (use_top_mask) {
      top_mask = top[1]->cpu_data();
    } else {
      mask = max_idx_.cpu_data();
    }

    for (int n = 0; n < top[0]->num(); ++n) {
    	// 
      for (int c = 0; c < pooled_channels_; ++c) {
        for (int ph = 0; ph < pooled_height_; ++ph) {
          for (int pw = 0; pw < pooled_width_; ++pw) {

            // const int index = ph * pooled_width_ + pw;
            const int index = (c * pooled_height_ + ph) * pooled_width_ + pw;
            // 
            const int bottom_index =
                use_top_mask ? top_mask[index] : mask[index];
            // 
            bottom_diff[bottom_index] += top_diff[index];

          }	// end pooled_width_
        }	// end pooled_height_
      }	// end pooled_channels_

      // 
      bottom_diff += bottom[0]->offset(1);
      top_diff += top[0]->offset(1);
      if (use_top_mask) {
        top_mask += top[0]->offset(1);
      } else {
        mask += top[0]->offset(1);
      }
    }	// end num_
    break;
  
  // Ave
  case GroupPoolingParameter_PoolMethod_AVE:
    // The main loop
    for (int n = 0; n < top[0]->num(); ++n) {
    	// 
      for (int c = 0; c < pooled_channels_; ++c) {
        for (int ph = 0; ph < pooled_height_; ++ph) {
          for (int pw = 0; pw < pooled_width_; ++pw) {

            int hstart = ph * stride_h_ - pad_h_;
            int wstart = pw * stride_w_ - pad_w_;
            int cstart = c  * temporal_stride_;

            int hend = min(hstart + kernel_h_, height_ + pad_h_);
            int wend = min(wstart + kernel_w_, width_ + pad_w_);
            int cend = min(cstart + temporal_length_, channels_);

            hstart = max(hstart, 0);
            wstart = max(wstart, 0);
            cstart = max (cstart, 0);

            hend = min(hend, height_);
            wend = min(wend, width_);
            cend = min(cend, channels_);

            // ((n * channels() + c) * height() + h) * width() + w;
            // const int pool_index = ph * pooled_width_ + pw;
            const int pool_index = (c * pooled_height_ + ph) * pooled_width_ + pw;

            // 
            int pool_size = (cend - cstart) * (hend - hstart) * (wend - wstart);

            for(int c2 = cstart; c2 < cend; ++c2) {
	            for (int h = hstart; h < hend; ++h) {
	              for (int w = wstart; w < wend; ++w) {
	              	// const int index = h * width_ + w;
	                const int index = (c2 * height_ + h) * width_ + w;

	                // set value
	                bottom_diff[index] += (top_diff[pool_index] / pool_size);
	              }
	            }
	          }

          }	// end pooled_width_
        }	// end pooled_height_
      }	// pooled_channels_

      // offset
      bottom_diff += bottom[0]->offset(1);
      top_diff += top[0]->offset(1);
    }	// end num_
    break;
  
  // Stochastic
  case GroupPoolingParameter_PoolMethod_STOCHASTIC:
    NOT_IMPLEMENTED;
    break;

  // default
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
}


#ifdef CPU_ONLY
STUB_GPU(Pooling3DLayer);
#endif


INSTANTIATE_CLASS(Pooling3DLayer);
REGISTER_LAYER_CLASS(Pooling3D);


}  // namespace caffe