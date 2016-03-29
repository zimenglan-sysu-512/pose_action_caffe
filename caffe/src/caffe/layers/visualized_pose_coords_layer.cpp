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

// if bottom.size() == 2:
//  bottom[0]: either predicted or ground truth
//  bottom[1]: aux info blob
// if bottom.size() == 3:
//  bottom[0]: predicted
//  bottom[1]: ground truth
//  bottom[2]: aux info blob
template <typename Dtype>
void VisualizedPoseCoordsLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
  CHECK(this->layer_param_.has_visual_pose_coords_param());
  const VisualPoseCoordsParameter visual_pose_coords_param =
      this->layer_param_.visual_pose_coords_param();
  CHECK(visual_pose_coords_param.has_coords_path());
  CHECK(visual_pose_coords_param.has_coords_files_name());
  CHECK(visual_pose_coords_param.has_coords_images_name());
  CHECK((visual_pose_coords_param.phase_name_size() == 1) || 
    (visual_pose_coords_param.phase_name_size() == 3));
  CHECK(visual_pose_coords_param.has_skel_path());
  CHECK(visual_pose_coords_param.has_is_draw_text());
  
  this->coords_path_ = 
      visual_pose_coords_param.coords_path();
  this->coords_files_name_ = 
      visual_pose_coords_param.coords_files_name();
  this->coords_images_name_ = 
      visual_pose_coords_param.coords_images_name();
  int pns = visual_pose_coords_param.phase_name_size();
  for(int pn = 0; pn < pns; pn++) {
    std::string phase_name = visual_pose_coords_param.phase_name(pn);
    this->phase_name_.push_back(phase_name);
  }
  this->is_draw_skel_ = visual_pose_coords_param.has_is_draw_skel() ? 
      visual_pose_coords_param.is_draw_skel() :
      false;
  this->skel_path_ = visual_pose_coords_param.skel_path();
  this->is_draw_text_ = visual_pose_coords_param.is_draw_text();
  // use default values
  this->img_ext_ = visual_pose_coords_param.img_ext();
  this->file_ext_ = visual_pose_coords_param.file_ext();

  this->coords_files_path_ = 
      this->coords_path_ + this->coords_files_name_;
  this->coords_images_path_ = 
      this->coords_path_ + this->coords_images_name_;
  
  CreateDir(this->coords_path_);
  CreateDir(this->coords_files_path_);
  CreateDir(this->coords_images_path_);
  // e.g. either `train/test` + or `train, test, fusion` 
  // (maybe you use use other name instead train/test/fusion)
  // must end with "/"
  for(int pn = 0; pn < this->phase_name_.size(); pn++) {
    const std::string file_phase_path = 
        this->coords_files_path_ + this->phase_name_[pn];
    const std::string image_phase_path = 
        this->coords_images_path_ + this->phase_name_[pn];
    CreateDir(file_phase_path);
    CreateDir(image_phase_path);
  }

  if(this->start_skel_idxs_.size() > 0) this->start_skel_idxs_.clear();
  if(this->end_skel_idxs_.size() > 0) this->end_skel_idxs_.clear();
  // index starts from zero
  get_skeleton_idxs(this->skel_path_, 
      this->start_skel_idxs_, this->end_skel_idxs_);
  CHECK(this->start_skel_idxs_.size() > 0);
  CHECK_EQ(this->start_skel_idxs_.size(), this->end_skel_idxs_.size());
  LOG(INFO);
  LOG(INFO) << "coords_path:" << this->coords_path_;
  LOG(INFO) << "coords_files_path:" << this->coords_files_path_;
  LOG(INFO) << "coords_images_path:" << this->coords_files_path_;
  LOG(INFO);
}

template <typename Dtype>
void VisualizedPoseCoordsLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
  this->num_ = bottom[0]->num();
  this->channels_ = bottom[0]->channels();
  this->height_ = bottom[0]->height();
  this->width_ = bottom[0]->width();
  CHECK_EQ(this->channels_, bottom[0]->count() / bottom[0]->num());

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
}

// if bottom.size() == 3, then use this order for bottom blobs: 
//    predicted, ground truth, aux info
template <typename Dtype>
void VisualizedPoseCoordsLayer<Dtype>::WriteFiles(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
  const int b_size = bottom.size();

  for(int item_id = 0; item_id < this->num_; item_id++) {
    // corresponding image name
    const std::string objidx = this->objidxs_[item_id];
    const std::string imgidx = this->imgidxs_[item_id];

    for(int bs = 0; bs < b_size - 1; bs++) {
      const std:: string file_path = this->coords_files_path_ + 
          this->phase_name_[bs] + imgidx + "_" + objidx + this->file_ext_;
      std::ofstream filer(file_path.c_str());
      CHECK(filer);
      for(int part_idx = 0; part_idx < this->channels_; part_idx += 2) {
        const Dtype x = bottom[bs]->data_at(item_id, part_idx + 0, 0, 0);
        const Dtype y = bottom[bs]->data_at(item_id, part_idx + 1, 0, 0);
        filer << x << " " << y << std::endl;
      }
    }
  }
}

template <typename Dtype>
void VisualizedPoseCoordsLayer<Dtype>::WriteImages(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
  const int b_size = bottom.size();
  const Dtype* aux_info = bottom[b_size - 1]->cpu_data();

  // default
  const int radius = 2;
  const int dx = 4;
  const int dy = 4;
  const int thickness = 2;
  const int fontFace = cv::FONT_HERSHEY_SIMPLEX;
  const float fontScale = .5;
  const cv::Scalar color1(216, 16, 216);
  const cv::Scalar color2(16, 216, 21);
  std::vector<cv::Scalar> colors;
  colors.push_back(color1);
  colors.push_back(color2);
  const cv::Scalar skel_color(16, 216, 21);
  const cv::Scalar text_color_ood(166, 12, 98);
  const cv::Scalar text_color_even(126, 112, 28);

  for(int item_id = 0; item_id < this->num_; item_id++) {
    // corresponding image name
    const std::string objidx = this->objidxs_[item_id];
    const std::string imgidx = this->imgidxs_[item_id];
    const std::string origin_image_path = this->images_paths_[item_id];
    cv::Mat img = cv::imread(origin_image_path);

    const int aux_info_offset = bottom[b_size - 1]->offset(item_id);
    // const Dtype img_ind     = aux_info[aux_info_offset + 0];
    const Dtype img_width   = aux_info[aux_info_offset + 1];
    const Dtype img_height  = aux_info[aux_info_offset + 2];
    // const Dtype im_scale    = aux_info[aux_info_offset + 3];
    const Dtype flippable   = aux_info[aux_info_offset + 4];
    if(flippable) {
      // >0: horizontal; <0: horizontal&vertical; =0: vertical
      const int flipCode = 1;
      cv::flip(img, img, flipCode);
    }

    CHECK_EQ(img.cols, img_width);
    CHECK_EQ(img.rows, img_height);
    // draw circle
    for(int bs = 0; bs < b_size - 1; bs++) {
      for(int idx = 0; idx < this->channels_; idx +=2) {
        // get point
        const Dtype x = bottom[bs]->data_at(item_id, idx, 0, 0);
        const Dtype y = bottom[bs]->data_at(item_id, idx + 1, 0, 0);
        cv::Point p(x, y);
        cv::circle(img, p, radius, colors[bs], thickness);

        if(this->is_draw_text_ && bs < 1) {
          const int idx2 = idx / 2;
          const std::string text = to_string(idx2);
          if(idx2 % 2) {
            cv::Point text_point(x - dx, y - dy);
            cv::putText(img, text, text_point, fontFace, fontScale, text_color_ood);
          } else {
            cv::Point text_point(x + dx, y + dy);
            cv::putText(img, text, text_point, fontFace, fontScale, text_color_even);
          }
        } // end if
      } // end idx
    } // end bs

    // draw skeleton -- only 
    if(this->is_draw_skel_) { 
      const int idx3 = b_size - 2;
      for(int s_idx = 0; s_idx < this->start_skel_idxs_.size(); s_idx++) {
        const int start_idx = this->start_skel_idxs_[s_idx];
        const int end_idx = this->end_skel_idxs_[s_idx];
        const int si = start_idx * 2;
        const int ei = end_idx * 2;
        // get point
        const Dtype sx = bottom[idx3]->data_at(item_id, si, 0, 0);
        const Dtype sy = bottom[idx3]->data_at(item_id, si + 1, 0, 0);
        const cv::Point sp(sx, sy);
        const Dtype ex = bottom[idx3]->data_at(item_id, ei, 0, 0);
        const Dtype ey = bottom[idx3]->data_at(item_id, ei + 1, 0, 0);
        const cv::Point ep(ex, ey);
        // 
        cv::line(img, sp, ep, skel_color, thickness);
      } // end s_idx
    }

    // save visual image
    std::string visual_image_path;
    if(b_size == 2) {
      visual_image_path = this->coords_images_path_ + 
          this->phase_name_[0] + imgidx + "_" + objidx + this->img_ext_;
    } else {
      visual_image_path = this->coords_images_path_ + 
          this->phase_name_[b_size - 1] + imgidx + "_" + objidx + this->img_ext_;
    }
    cv::imwrite(visual_image_path, img);
  } // end item_id
}

template <typename Dtype>
void VisualizedPoseCoordsLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
  this->objidxs_ = GlobalVars::objidxs();
  this->imgidxs_ = GlobalVars::imgidxs();
  this->images_paths_ = GlobalVars::images_paths();
  CHECK(this->imgidxs_.size() > 0);
  CHECK_EQ(this->imgidxs_.size(), this->num_);
  CHECK_EQ(this->imgidxs_.size(), this->objidxs_.size());
  CHECK_EQ(this->imgidxs_.size(), this->images_paths_.size());
  
  // use default value
  if(this->layer_param_.visual_pose_coords_param().is_write_file()) {
    this->WriteFiles(bottom, top);
  } 
  if(this->layer_param_.visual_pose_coords_param().is_write_image()) {
    this->WriteImages(bottom, top);
  }
}

template <typename Dtype>
void VisualizedPoseCoordsLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, 
    const vector<Blob<Dtype>*>& bottom) 
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
STUB_GPU(VisualizedPoseCoordsLayer);
#endif

INSTANTIATE_CLASS(VisualizedPoseCoordsLayer);
REGISTER_LAYER_CLASS(VisualizedPoseCoords);

}  // namespace caffe