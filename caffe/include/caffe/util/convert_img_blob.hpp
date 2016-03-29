#ifndef CAFFE_UTIL_CONVERT_IMG_BLOB_H_
#define CAFFE_UTIL_CONVERT_IMG_BLOB_H_

#include <unistd.h>
#include <string>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "caffe/common.hpp"
#include "caffe/blob.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/proto/caffe.pb.h"

using cv::Point_;
using cv::Mat_;
using cv::Mat;
using cv::vector;

namespace caffe {

cv::Mat ImageRead(const string& filename, const bool is_color = true);

cv::Mat ResizeImage(const cv::Mat im, const float& scale);

cv::Mat ResizeImage(const std::string& im_path, const int& ow, const int& oh, 
										const float& scale, const bool is_color);
										
bool ResizeImage(const std::string& im_path, cv::Mat& im2, const int& ow, 
								 const int& oh, const float& scale, const bool is_color);

}  // namespace caffe

#endif   // CAFFE_UTIL_CONVERT_IMG_BLOB_H_
