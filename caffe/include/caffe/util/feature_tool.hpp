#ifndef CAFFE_UTIL_FEATURE_TOOL_H_
#define CAFFE_UTIL_FEATURE_TOOL_H_

#include <vector>
#include <opencv2/core/core.hpp>

#include "caffe/blob.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/util_img.hpp"
#include "caffe/util/math_functions.hpp"

using cv::Point_;
using cv::Mat_;
using cv::Mat;
// using cv::vector;

namespace caffe {

template <typename Dtype>
cv::Mat BlobToGreyImage(const Blob<Dtype>* blob, 
		const int n, const int c, 
		const Dtype scale = Dtype(1.0));

template <typename Dtype>
void GetHOGFeature(
		const cv::Mat& img, 
    Blob<Dtype>* features, 
		const int n, 
    int num_orinetations = 24,
    int num_half_orinetations = 12,  
    int block_size = 5,
    const float eps = 0.0001f);

template <typename Dtype>
void GetGradientFeature(
		const cv::Mat& img, 
		cv::Mat& feature, 
		const bool only_gradient = false);

template<typename Dtype>
void GetGradientFeature(
    const Datum& datum, 
    Dtype* gradient_data, 
    const int gradient_offset,
    const int height,
    const int width,
    const bool only_gradient = false 
    /* true: v, dx, dy, false: v */) ;

template<typename Dtype>
void GetGradientFeature(
    const Dtype* image_data, 
    const int image_data_offset,
    Dtype* gradient_data, 
    const int gradient_offset,
    const int channels,
    const int height,
    const int width,
    const bool only_gradient = false 
    /* true: v, dx, dy, false: v */);

}  // namespace caffe

#endif   // CAFFE_UTIL_FEATURE_TOOL_H_