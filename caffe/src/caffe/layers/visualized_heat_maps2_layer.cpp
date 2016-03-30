#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

#include "caffe/layer.hpp"
#include "caffe/common.hpp"
#include "caffe/pose_estimation_layers.hpp"
#include "caffe/util/pose_tool.hpp"
#include "caffe/global_variables.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

// if bottom.size() == 1:
//  bottom[0]: either predicted or ground truth
// if bottom.size() == 2:
//  bottom[0]: predicted
//  bottom[1]: ground truth
template <typename Dtype>
void VisualizedHeatMaps2Layer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) 
{
  CHECK(this->layer_param_.has_visual_heat_maps_param());
  const VisualHeatMapsParameter visual_heat_maps_param =
      this->layer_param_.visual_heat_maps_param();
  CHECK(visual_heat_maps_param.has_heat_map_path());
  CHECK(visual_heat_maps_param.has_heat_map_files_name());
  CHECK(visual_heat_maps_param.has_heat_map_images_name());
  CHECK(visual_heat_maps_param.has_visual_type());
  CHECK(visual_heat_maps_param.has_threshold());
  CHECK(visual_heat_maps_param.has_phase_name());

  this->heat_map_path_ = visual_heat_maps_param.heat_map_path();
  this->heat_map_files_name_ = visual_heat_maps_param.heat_map_files_name();
  this->heat_map_images_name_ = visual_heat_maps_param.heat_map_images_name();
  this->visual_type_ = visual_heat_maps_param.visual_type();
  this->threshold_ = Dtype(visual_heat_maps_param.threshold());
  this->phase_name_ = visual_heat_maps_param.phase_name();
  // use default values
  this->img_ext_ = visual_heat_maps_param.img_ext();
  this->file_ext_ = visual_heat_maps_param.file_ext();

  CHECK_GE(this->visual_type_, 0);
  CHECK_LE(this->visual_type_, 2);
  CHECK_GE(this->threshold_, Dtype(0.0));
  CHECK_LE(this->threshold_, Dtype(1.0));

  this->heat_map_files_path_ = this->heat_map_path_ + this->heat_map_files_name_;
  this->heat_map_images_path_ = this->heat_map_path_ + this->heat_map_images_name_;

  CreateDir(this->heat_map_path_);
  CreateDir(this->heat_map_files_path_);
  CreateDir(this->heat_map_images_path_);
}

template <typename Dtype>
void VisualizedHeatMaps2Layer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) 
{
  this->num_ = bottom[0]->num();
  this->channels_ = bottom[0]->channels();
  this->height_ = bottom[0]->height();
  this->width_ = bottom[0]->width();
  this->heat_num_ = this->height_ * this->width_;
  
  for(int idx = 0; idx < bottom.size() - 1; idx++) {
    CHECK_EQ(this->num_, bottom[0]->num());
    CHECK_EQ(this->channels_, bottom[0]->channels());
    CHECK_EQ(this->height_, bottom[0]->height());
    CHECK_EQ(this->width_, bottom[0]->width());
  }

  // aux info (img_ind, width, height, im_scale, flippable)
  const int b_size = bottom.size();
  CHECK_EQ(bottom[0]->num(), bottom[b_size - 1]->num());
  CHECK_EQ(bottom[b_size - 1]->channels(), 5);
  CHECK_EQ(bottom[b_size - 1]->height(), 1);
  CHECK_EQ(bottom[b_size - 1]->width(), 1);
  Dtype max_width = Dtype(-1);
  Dtype max_height = Dtype(-1);
  const Dtype* aux_info = bottom[b_size - 1]->cpu_data();
  for(int item_id = 0; item_id < this->num_; item_id++) {
    /// (img_ind, width, height, im_scale, flippable)
    const int offset = bottom[b_size - 1]->offset(item_id);
    /// const Dtype img_ind   = aux_info[offset + 0];
    const Dtype width     = aux_info[offset + 1];
    const Dtype height    = aux_info[offset + 2];
    const Dtype im_scale  = aux_info[offset + 3];
    /// const Dtype flippable = aux_info[offset + 4];
    max_width = std::max(max_width, width * im_scale);
    max_height = std::max(max_height, height * im_scale);
  }
  // reshape & reset
  this->max_width_ = max_width;
  this->max_height_ = max_height;
}

// if bottom.size() == 2, then use this order for bottom blobs: 
//    predicted, ground truth
template <typename Dtype>
void VisualizedHeatMaps2Layer<Dtype>::WriteFiles(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) 
{
  this->phase_path_ = this->heat_map_files_path_ + this->phase_name_;
  CreateDir(this->phase_path_);
  for(int item_id = 0; item_id < this->num_; item_id++) {
    // corresponding image name
    const std::string objidx = this->objidxs_[item_id];
    const std::string imgidx = this->imgidxs_[item_id];

    for(int part_idx = 0; part_idx < this->channels_; part_idx++) {
      // get file handler
      const std::string file_path = this->phase_path_ + imgidx + "_" + objidx
          + "_" + to_string(part_idx) + this->file_ext_;
      std::ofstream filer(file_path.c_str());
      CHECK(filer);

      // write results
      for(int bs = 0; bs < bottom.size() - 1; bs++) {
        for(int h = 0; h < this->height_; ++h) {
          for(int w = 0; w < this->width_; ++w) {
            filer << bottom[bs]->data_at(item_id, part_idx, h, w) << " ";
          }
          filer << std::endl;
        }
        filer << std::endl;
        if(bs != bottom.size() - 2) {
          filer << GlobalVars::SpiltCodeBoundWithStellate();
          filer << std::endl;
        }
      }

      // close
      filer.close();
    }
  }
}

template <typename Dtype>
void VisualizedHeatMaps2Layer<Dtype>::WriteImages(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) 
{
  // variables
  const cv::Vec3b color1(216, 16, 216);
  const cv::Vec3b color2(16, 196, 21);
  const cv::Vec3b color3(166, 196, 21);
  const cv::Scalar color4(21, 16, 162);
  const cv::Scalar color5(12, 61, 162);

  this->phase_path_ = this->heat_map_images_path_ + this->phase_name_;
  CreateDir(this->phase_path_);

  // deal with each image
  const Dtype rsw = this->width_  / Dtype(this->max_width_ + 0.0);
  const Dtype rsh = this->height_ / Dtype(this->max_height_ + 0.0);
  const int b_size = bottom.size() - 2;
  const Dtype* aux_info = bottom[b_size + 1]->cpu_data();
  for(int item_id = 0; item_id < this->num_; item_id++) {
    // corresponding image name
    const std::string objidx = this->objidxs_[item_id];
    const std::string imgidx = this->imgidxs_[item_id];
    const std::string origin_image_path = this->images_paths_[item_id];
    const int aux_info_offset = bottom[b_size + 1]->offset(item_id);
    // const Dtype img_ind     = aux_info[aux_info_offset + 0];
    const Dtype img_width   = aux_info[aux_info_offset + 1];
    const Dtype img_height  = aux_info[aux_info_offset + 2];
    const Dtype im_scale    = aux_info[aux_info_offset + 3];
    const Dtype flippable   = aux_info[aux_info_offset + 4];
    const int width  = (img_width  * im_scale) * rsw;
    const int height = (img_height * im_scale) * rsh;
    cv::Mat img = cv::imread(origin_image_path);
    CHECK(img.data) << " Loading " << origin_image_path << " failed.";

    cv::resize(img, img, cv::Size(width, height));
    if(flippable) {
      // >0: horizontal; <0: horizontal&vertical; =0: vertical
      const int flipCode = 1;
      cv::flip(img, img, flipCode);
    }

    CHECK_LE(img.cols, this->width_);
    CHECK_LE(img.rows, this->height_);

    for(int c = 0; c < this->channels_; c++) {
      cv::Mat img2 = img.clone();
      Dtype cx1 = Dtype(-1);
      Dtype cy1 = Dtype(-1);
      Dtype ms1 = Dtype(-FLT_MAX);
      // Dtype cx2 = Dtype(-1);
      // Dtype cy2 = Dtype(-1);
      // Dtype ms2 = Dtype(-FLT_MAX);
      for(int h = 0; h < img.rows; h++) {
        for(int w = 0; w < img.cols; w++) { 
          if(b_size) {
            const Dtype v1 = bottom[b_size - 1]->data_at(item_id, c, h, w);
            const Dtype v2 = bottom[b_size - 0]->data_at(item_id, c, h, w);
            if (v1 > this->threshold_ && v2 > this->threshold_) {
              // set pixel
              img2.at<cv::Vec3b>(h, w) = color3;
            } else if(v1 > this->threshold_) {
              img2.at<cv::Vec3b>(h, w) = v1 * color1;
            } else if(v2 > this->threshold_) {
              img2.at<cv::Vec3b>(h, w) = v2 * color2;
            }

            if(v1 > ms1) {
              cx1 = w; cy1 = h; ms1 = v1;
            }
            // if(v2 > ms2) {
            //   cx2 = w; cy2 = h; ms2 = v2;
            // }
          } else {
            const Dtype v = bottom[b_size]->data_at(item_id, c, h, w);
            if (v > this->threshold_) {
              // set pixel
              img2.at<cv::Vec3b>(h, w) = v * color1;
            }
            if(v > ms1) {
              cx1 = w; cy1 = h; ms1 = v;
            }
          }
        }
      }  
      cv::Point p1(cx1, cy1);
      cv::circle(img2, p1, 2, color4, 2);
      // if(b_size) {
      //   cv::Point p2(cx2, cy2);
      //   cv::circle(img2, p2, 2, color5, 2);
      // }
      // save
      const std::string image_path = this->phase_path_ + imgidx + "_" + 
          objidx + "_" + to_string(c) + this->img_ext_;
      const std::string dire = DireName(image_path);
      CreateDir(dire.c_str(), 0);
      cv::imwrite(image_path, img2);
    }
  }
}

template <typename Dtype>
void VisualizedHeatMaps2Layer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) 
{
  this->objidxs_ = GlobalVars::objidxs();
  this->imgidxs_ = GlobalVars::imgidxs();
  this->images_paths_ = GlobalVars::images_paths();
  CHECK(this->imgidxs_.size() > 0);
  CHECK_EQ(this->imgidxs_.size(), this->num_);
  CHECK_EQ(this->imgidxs_.size(), this->objidxs_.size());
  CHECK_EQ(this->imgidxs_.size(), this->images_paths_.size());
  
  if(this->visual_type_ == 0) {
    this->WriteFiles(bottom, top);
  } else if(this->visual_type_ == 1) {
    this->WriteImages(bottom, top);
  } else if(this->visual_type_ == 2) {
    this->WriteFiles(bottom, top);
    this->WriteImages(bottom, top);
  } else {
    // NOT_IMPLEMENTED;
    LOG(INFO) << "Do nothing...";
  }
}

template <typename Dtype>
void VisualizedHeatMaps2Layer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) 
{
  const Dtype Zero = Dtype(0);
  CHECK_EQ(propagate_down.size(), bottom.size());

  for (int i = 0; i < propagate_down.size(); ++i) {
    if (propagate_down[i]) { 
      // NOT_IMPLEMENTED; 
      caffe_set(bottom[i]->count(), Zero, bottom[i]->mutable_cpu_diff());
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(VisualizedHeatMaps2Layer);
#endif

INSTANTIATE_CLASS(VisualizedHeatMaps2Layer);
REGISTER_LAYER_CLASS(VisualizedHeatMaps2);

}  // namespace caffe