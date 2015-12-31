// CopyRight by Dengke Dong, 2015-12-30

#include <opencv2/core/core.hpp>

#include <fstream>   // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "boost/algorithm/string.hpp"
#include "google/protobuf/text_format.h"

#include "caffe/layer.hpp"
#include "caffe/util/util_img.hpp"
#include "caffe/util/util_img.hpp"
#include "caffe/util/pose_tool.hpp"
#include "caffe/global_variables.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/pose_estimation_layers.hpp"

#define __LOAD_DATA_IMAGE_LAYER_VISUAL__
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

namespace caffe {

template <typename Dtype>
void LoadImageDataLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
  const LoadDataImageParameter load_data_image_param = 
  		this->layer_param_.load_data_image_param();
  img_ext_    		 = load_data_image_param.img_ext();
  is_color_    		 = load_data_image_param.is_color();
  root_folder_ 		 = load_data_image_param.root_folder();
  visual_path_ 		 = load_data_image_param.visual_path();
  has_visual_path_ = load_data_image_param.has_visual_path();
  if(has_visual_path_) {
  	CreateDir(visual_path_.c_str(), 0);
  }
}

// aux_info blob produced by data layer (RandomImageData2Layer)
template <typename Dtype>
void LoadImageDataLayer<Dtype>::Reshape(
		const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top) 
{
	int num      = bottom[0]->num();
  int count    = bottom[0]->count();
  int channels = bottom[0]->channels();
  CHECK_EQ(channels, 5);
  CHECK_EQ(channels, count / num);

  Dtype max_width  = Dtype(-1);
  Dtype max_height = Dtype(-1);
  const Dtype* aux_info = bottom[0]->cpu_data();

   // (img_ind, width, height, im_scale, flippable)
  for(int n_idx = 0; n_idx < num; n_idx++) {
    const int offset = bottom[0]->offset(n_idx);
    const Dtype width     = aux_info[offset + 1];
    const Dtype height    = aux_info[offset + 2];
    const Dtype im_scale  = aux_info[offset + 3];
    const Dtype r_width   = width  * im_scale;
    const Dtype r_height  = height * im_scale;
    max_width  = std::max(max_width,  r_width);
    max_height = std::max(max_height, r_height);
  }
  // Reshape  
  int width  = int(max_width);
  int height = int(max_height);
  int n_channels = is_color_ ? 3 : 1;
  if(height <= 0 || width <= 0) {
    // You must know where to set `g_width` and `g_height`
    // Just for the initialization, like the deploy.prototxt
    //    in tools/camera_pose.cpp
    // The shape must keep the same as input layer
    const int g_width  = GlobalVars::g_width();
    const int g_height = GlobalVars::g_height();
    top[0]->Reshape(num, n_channels, g_width, g_height);
  } else {
    top[0]->Reshape(num, n_channels, height, width);
  }
}

// This function is used to create a thread that prefetches the data.
template <typename Dtype>
void LoadImageDataLayer<Dtype>::load_data_image2blob(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
	int num 		 = top[0]->num();
	int count    = top[0]->count();

	const std::vector<std::string> objidxs = GlobalVars::objidxs();
	const std::vector<std::string> imgidxs = GlobalVars::imgidxs();
	CHECK_EQ(objidxs.size(), num);
	CHECK_EQ(imgidxs.size(), num);

	const Dtype* aux_info = bottom[0]->cpu_data();
	Dtype* data_blob 			= top[0]->mutable_cpu_data();
	caffe_set(count, Dtype(0), data_blob);

	int width, height;
	Dtype r_width, r_height;
	Dtype im_width, im_height;
	Dtype im_scale, im_flipped;
	std::string imgidx, im_path;

	 // (im_ind, width, height, im_scale, flippable)
	for(int n_idx = 0; n_idx < num; n_idx++) {
	  const int offset = bottom[0]->offset(n_idx);
	  im_width   = aux_info[offset + 1];
	  im_height  = aux_info[offset + 2];
	  im_scale   = aux_info[offset + 3];
	  im_flipped = aux_info[offset + 4];
	  r_width  	 = im_width  * im_scale;
	  r_height 	 = im_height * im_scale;

	  // image path (person mask, motion map or other thing...)
	  imgidx  = imgidxs[n_idx];
	  im_path = root_folder_ + imgidx + img_ext_;
	  // 3 channels or 1 channel
	  int flag = (is_color_ ? CV_LOAD_IMAGE_COLOR : CV_LOAD_IMAGE_GRAYSCALE);
	  // read 
	  cv::Mat im = cv::imread(im_path, flag);
	  if (!im.data) {
	    LOG(ERROR) << "1 Could not open or find file " << im_path;
	    return;
	  }
	  // check whether match the size
	  CHECK_EQ(im_width,  im.cols) << "does not match the width (size) of input image";
	  CHECK_EQ(im_height, im.rows) << "does not match the height (size) of input image";

	  cv::Mat im2;
	  width  = int(r_width);
	  height = int(r_height);
	  cv::resize(im, im2, cv::Size(width, height));
	  CHECK(im2.data) << "2 Could not open or find file " << im_path;

	  if(im_flipped) {
	    // >0: horizontal; <0: horizontal&vertical; =0: vertical
	    const int flipCode = 1;
	    cv::flip(im2, im2, flipCode);
	  }

	  // 把图片拷贝到放在blob的右上角
	  ImageDataToBlob(top[0], n_idx, im2);

	  // visualization
	  #ifdef __LOAD_DATA_IMAGE_LAYER_VISUAL__
	  {
	  	if(!has_visual_path_) {
	  		continue;
	  	}
      cv::Mat im3, im4, im5;
      im3 = BlobToColorImage(top[0], n_idx);
      im4 = im3(cv::Rect(0, 0, width, height)).clone();
      int width2  = int(im_width);
      int height2 = int(im_height);
      cv::resize(im4, im5, cv::Size(width2, height2));
      const std::string im_path2 = visual_path_ + imgidx + img_ext_;
      LOG(INFO) << "visualized image path: " << im_path2;
      cv::imwrite(im_path2, im5);

      // cv::imshow(imgidx, im3);
      // cv::waitKey(0);
      // cv::imshow(imgidx, im4);
      // cv::waitKey(0);
      // cv::imshow(imgidx, im5);
      // cv::waitKey(0);
      // cv::destroyAllWindows();
    }
	  #endif
	}
}

template <typename Dtype>
void LoadImageDataLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
	load_data_image2blob(bottom, top);
}

#ifdef CPU_ONLY
STUB_GPU(LoadImageDataLayer);
#endif

INSTANTIATE_CLASS(LoadImageDataLayer);
REGISTER_LAYER_CLASS(LoadImageData);

}  // namespace caffe
