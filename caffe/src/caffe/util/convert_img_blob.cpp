#include <fcntl.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdint.h>

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/convert_img_blob.hpp"

namespace caffe {

cv::Mat ImageRead(const string& filename, const bool is_color) {
  int flag   = is_color ? CV_LOAD_IMAGE_COLOR 
  										  : CV_LOAD_IMAGE_GRAYSCALE;
  cv::Mat im = cv::imread(filename, flag);
  if (!im.data) {
    LOG(ERROR) << "Could not open or find file " << filename;
  }
  return im;
}

cv::Mat ResizeImage(const cv::Mat im, const float& scale) {
	cv::Mat im2;
	if (!im.data) {
    LOG(ERROR) << "\nEmpty image...\n";
    return im2;
  }

  const int w = im.cols;	// w
  const int h = im.rows;	// h
  // const int c = im.channels();

  const int w2 = int(w * scale);
  const int h2 = int(h * scale);

  // resize
  cv::resize(im, im2, cv::Size(w2, h2));

  if(!im2.data) {
  	LOG(ERROR) << "\nEmpty resized image...\n";
  }
  return im2;
}

cv::Mat ResizeImage(const std::string& im_path, const int& ow, const int& oh, 
										const float& scale, const bool is_color) 
{
	int flag   = is_color ? CV_LOAD_IMAGE_COLOR 
											  : CV_LOAD_IMAGE_GRAYSCALE;
	// imread											  
	cv::Mat im = cv::imread(im_path, flag);

	cv::Mat im2;
	if (!im.data) {
    LOG(ERROR) << "\nEmpty image...\n";
    return im2;
  }

  const int w = im.cols;	// w
  const int h = im.rows;	// h
  // const int c = im.channels();

  if(ow <= 0 || oh <= 0) {
  	CHECK_EQ(w, ow) << "does not match the w to ow from " << im_path;
  	CHECK_EQ(h, oh) << "does not match the h to ow from " << im_path;
  }

  const int w2 = int(w * scale);
  const int h2 = int(h * scale);

  // resize
  cv::resize(im, im2, cv::Size(w2, h2));

  if(!im2.data) {
  	LOG(ERROR) << "\nEmpty resized image...\n";
  }
  return im2;
}

bool ResizeImage(const std::string& im_path, cv::Mat& im2, const int& ow, 
								 const int& oh, const float& scale, const bool is_color) 
{
	int flag   = is_color ? CV_LOAD_IMAGE_COLOR 
											  : CV_LOAD_IMAGE_GRAYSCALE;
	// imread											  
	cv::Mat im = cv::imread(im_path, flag);

	if (!im.data) {
    LOG(INFO) << "\nEmpty image...\n";
    return false;
  }

  const int w = im.cols;	// w
  const int h = im.rows;	// h
  // const int c = im.channels();

  if(ow <= 0 || oh <= 0) {
  	CHECK_EQ(w, ow) << "does not match the w to ow from " << im_path;
  	CHECK_EQ(h, oh) << "does not match the h to ow from " << im_path;
  }

  const int w2 = int(w * scale);
  const int h2 = int(h * scale);

  // resize
  cv::resize(im, im2, cv::Size(w2, h2));

  if(!im2.data) {
  	LOG(INFO) << "\nEmpty resized image...\n";
  	return false;
  }

  return true;
}

}  // namespace caffe
