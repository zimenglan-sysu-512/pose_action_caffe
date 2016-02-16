// ------------------------------------------------------------------
// Fast R-CNN
// Copyright (c) 2015 Microsoft
// Licensed under The MIT License [see fast-rcnn/LICENSE for details]
// Written by Ross Girshick
// 
// Modified by DDK
// ------------------------------------------------------------------

#include <cfloat>

#include "caffe/fast_rcnn_action_layers.hpp"

using std::max;
using std::min;
using std::floor;
using std::ceil;

namespace caffe {

template <typename Dtype>
void ScenePoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) 
{
  ScenePoolingParameter scene_pooling_param = 
      this->layer_param_.scene_pooling_param();
  CHECK_GT(scene_pooling_param.pooled_h(), 0)
      << "pooled_h must be > 0";
  CHECK_GT(scene_pooling_param.pooled_w(), 0)
      << "pooled_w must be > 0";
  // 
  pooled_height_ = scene_pooling_param.pooled_h();
  pooled_width_  = scene_pooling_param.pooled_w();
}



template <typename Dtype>
void ScenePoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) 
{
  num_      = bottom[0]->num();
  channels_ = bottom[0]->channels();
  height_   = bottom[0]->height();
  width_    = bottom[0]->width();
  // Reshape
  top[0]->Reshape( num_, channels_, pooled_height_, pooled_width_);
  max_idx_.Reshape(num_, channels_, pooled_height_, pooled_width_);

  // where [img_ind, x1, y1, x2, y2]
  const int rois_channels = 5;
  rois_blob_.Reshape(num_, rois_channels, 1, 1);

  int rois_offset = 0;
  for(int idx = 0; idx < num_; idx++) {
    rois_blob_.mutable_cpu_data()[rois_offset++] = Dtype(idx);
    rois_blob_.mutable_cpu_data()[rois_offset++] = Dtype(0);
    rois_blob_.mutable_cpu_data()[rois_offset++] = Dtype(0);
    rois_blob_.mutable_cpu_data()[rois_offset++] = Dtype(width_  - 1);
    rois_blob_.mutable_cpu_data()[rois_offset++] = Dtype(height_ - 1);
  }
  CHECK_EQ(rois_blob_.count(), rois_offset);
  spatial_scale_ = Dtype(1.);
}



template <typename Dtype>
void ScenePoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) 
{
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_rois = rois_blob_.cpu_data();
  // Number of ROIs
  int num_rois = rois_blob_.num();
  int batch_size = bottom[0]->num();
  int top_count = top[0]->count();
  Dtype* top_data = top[0]->mutable_cpu_data();
  caffe_set(top_count, Dtype(-FLT_MAX), top_data);
  int* argmax_data = max_idx_.mutable_cpu_data();
  caffe_set(top_count, -1, argmax_data);

  // For each ROI R = [batch_index x1 y1 x2 y2]: max pool over R
  for (int n = 0; n < num_rois; ++n) {
    int roi_batch_ind = bottom_rois[0];
    int roi_start_w = round(bottom_rois[1] * spatial_scale_);
    int roi_start_h = round(bottom_rois[2] * spatial_scale_);
    int roi_end_w   = round(bottom_rois[3] * spatial_scale_);
    int roi_end_h   = round(bottom_rois[4] * spatial_scale_);
    CHECK_GE(roi_batch_ind, 0);
    CHECK_LT(roi_batch_ind, batch_size);

    int roi_height = max(roi_end_h - roi_start_h + 1, 1);
    int roi_width = max(roi_end_w - roi_start_w + 1, 1);
    const Dtype bin_size_h = static_cast<Dtype>(roi_height)
                             / static_cast<Dtype>(pooled_height_);
    const Dtype bin_size_w = static_cast<Dtype>(roi_width)
                             / static_cast<Dtype>(pooled_width_);

    const Dtype* batch_data = bottom_data + bottom[0]->offset(roi_batch_ind);

    for (int c = 0; c < channels_; ++c) {
      for (int ph = 0; ph < pooled_height_; ++ph) {
        for (int pw = 0; pw < pooled_width_; ++pw) {
          // Compute pooling region for this output unit:
          //  start (included) = floor(ph * roi_height / pooled_height_)
          //  end (excluded) = ceil((ph + 1) * roi_height / pooled_height_)
          int hstart = static_cast<int>(floor(static_cast<Dtype>(ph)
                                              * bin_size_h));
          int wstart = static_cast<int>(floor(static_cast<Dtype>(pw)
                                              * bin_size_w));
          int hend = static_cast<int>(ceil(static_cast<Dtype>(ph + 1)
                                           * bin_size_h));
          int wend = static_cast<int>(ceil(static_cast<Dtype>(pw + 1)
                                           * bin_size_w));

          hstart = min(max(hstart + roi_start_h, 0), height_);
          hend = min(max(hend + roi_start_h, 0), height_);
          wstart = min(max(wstart + roi_start_w, 0), width_);
          wend = min(max(wend + roi_start_w, 0), width_);

          bool is_empty = (hend <= hstart) || (wend <= wstart);

          const int pool_index = ph * pooled_width_ + pw;
          if (is_empty) {
            top_data[pool_index] = 0;
            argmax_data[pool_index] = -1;
          }

          for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
              const int index = h * width_ + w;
              if (batch_data[index] > top_data[pool_index]) {
                top_data[pool_index] = batch_data[index];
                argmax_data[pool_index] = index;
              }
            }
          }
        }
      }
      // Increment all data pointers by one channel
      batch_data += bottom[0]->offset(0, 1);
      top_data += top[0]->offset(0, 1);
      argmax_data += max_idx_.offset(0, 1);
    }
    // Increment ROI data pointer
    bottom_rois += rois_blob_.offset(1);
  }
  // 
}


// template <typename Dtype>
// void ScenePoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
//       const vector<Blob<Dtype>*>& top) 
// {
//   // Number of ROIs
//   int batch_size = bottom[0]->num();
//   int top_count = top[0]->count();
//   // top data
//   Dtype* top_data = top[0]->mutable_cpu_data();
//   caffe_set(top_count, Dtype(-FLT_MAX), top_data);
//   // max indices
//   int* argmax_data = max_idx_.mutable_cpu_data();
//   caffe_set(top_count, -1, argmax_data);
//   // Get the feature maps of whole image from previous layer
//   const Dtype* bottom_data = bottom[0]->cpu_data();


//   // SPP for the whole images
//   for (int n = 0; n < batch_size; ++n) {
//     const int roi_start_w = 0;
//     const int roi_start_h = 0;
//     const int roi_end_w = width_ - 1;
//     const int roi_end_h = height_ - 1;
//     // 
//     const int roi_height = max(roi_end_h - roi_start_h + 1, 1);
//     const int roi_width = max(roi_end_w - roi_start_w + 1, 1);
//     // 
//     const Dtype bin_size_h = 
//         static_cast<Dtype>(roi_height) / static_cast<Dtype>(pooled_height_);
//     const Dtype bin_size_w = 
//         static_cast<Dtype>(roi_width) / static_cast<Dtype>(pooled_width_);
//     // 
//     for (int c = 0; c < channels_; ++c) {
//       // One feature map
//       for (int ph = 0; ph < pooled_height_; ++ph) {
//         for (int pw = 0; pw < pooled_width_; ++pw) {
//           // Compute whole image for this output unit:
//           //  start (included) = floor(ph * roi_height / pooled_height_)
//           //  end (excluded) = ceil((ph + 1) * roi_height / pooled_height_)
//           int hstart = 
//               static_cast<int>(floor(static_cast<Dtype>(ph) * bin_size_h));
//           int wstart = 
//               static_cast<int>(floor(static_cast<Dtype>(pw) * bin_size_w));
//           int hend = 
//               static_cast<int>(ceil(static_cast<Dtype>(ph + 1) * bin_size_h));
//           int wend = 
//               static_cast<int>(ceil(static_cast<Dtype>(pw + 1) * bin_size_w));
//           // Check the bound for start or end of width and height
//           hstart = min(max(hstart + roi_start_h, 0), height_);
//           hend = min(max(hend + roi_start_h, 0), height_);
//           wstart = min(max(wstart + roi_start_w, 0), width_);
//           wend = min(max(wend + roi_start_w, 0), width_);
//           // Check the valid cell of one pooling unit
//           bool is_empty = (hend <= hstart) || (wend <= wstart);
//           // Get the pooling index
//           const int pool_index = ph * pooled_width_ + pw;
//           // 
//           if (is_empty) {
//             top_data[pool_index] = 0;
//             argmax_data[pool_index] = -1;
//           }
//           // Max operation
//           for (int h = hstart; h < hend; ++h) {
//             for (int w = wstart; w < wend; ++w) {
//               const int index = h * width_ + w;
//               if (bottom_data[index] > top_data[pool_index]) {
//                 top_data[pool_index] = bottom_data[index];
//                 argmax_data[pool_index] = index;
//               }
//             }
//           }
//         } // end pooled_width_
//       } // end pooled_height_

//       // Increment all data pointers by one channel
//       bottom_data += bottom[0]->offset(0, 1);
//       top_data += top[0]->offset(0, 1);
//       argmax_data += max_idx_.offset(0, 1);
//     } // end channels_
//   } // end num_
// }



template <typename Dtype>
void ScenePoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) 
{
  NOT_IMPLEMENTED;
}


#ifdef CPU_ONLY
STUB_GPU(ScenePoolingLayer);
#endif

INSTANTIATE_CLASS(ScenePoolingLayer);
REGISTER_LAYER_CLASS(ScenePooling);

}  // namespace caffe
