#include <vector>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/core/core.hpp>

#include "caffe/layer.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/pose_tool.hpp"
#include "caffe/global_variables.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/pose_estimation_layers.hpp"

namespace caffe {

template <typename Dtype>
void CoordsToBboxesMasksLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
  CHECK(this->layer_param_.has_coord_to_bbox_masks_param());
  const CoordsToBboxMasksParameter coord_to_bbox_masks_param = 
      this->layer_param_.coord_to_bbox_masks_param();
  CHECK(coord_to_bbox_masks_param.has_top_id());
  CHECK(coord_to_bbox_masks_param.has_top_id2());
  CHECK(coord_to_bbox_masks_param.has_bottom_id());
  CHECK(coord_to_bbox_masks_param.has_bottom_id2());
  CHECK(coord_to_bbox_masks_param.has_value()); 

  this->top_id_         = coord_to_bbox_masks_param.top_id();
  this->top_id2_        = coord_to_bbox_masks_param.top_id2();
  this->bottom_id_      = coord_to_bbox_masks_param.bottom_id();
  this->bottom_id2_     = coord_to_bbox_masks_param.bottom_id2();
  this->value_ = Dtype(coord_to_bbox_masks_param.value());

  this->perb_num_    = 0;
  if(coord_to_bbox_masks_param.has_perturb_num()) {
    this->perb_num_  = coord_to_bbox_masks_param.perturb_num();
  }
  this->is_perturb_     = this->perb_num_ > 0;

  this->has_visual_path_ = coord_to_bbox_masks_param.has_visual_path();
  if(this->has_visual_path_) {
    this->visual_path_ = coord_to_bbox_masks_param.visual_path();  
    CreateDir(this->visual_path_.c_str(), 0);  
    CHECK(coord_to_bbox_masks_param.has_img_ext()); 
    this->img_ext_ = coord_to_bbox_masks_param.img_ext();
  }
  
  this->whole_ = false;
  const bool bo1 = this->bottom_id_ < 0 || this->top_id_ < 0;
  const bool bo2 = this->bottom_id2_ < 0 ||  this->top_id2_ < 0;
  const bool bo3 = this->bottom_id_ == this->top_id_;
  const bool bo4 = this->bottom_id2_ == this->top_id2_;
  if(bo1 || bo2 || bo3 || bo4) {
    this->top_id_ = 0;
    this->top_id2_ = 0;
    this->bottom_id_ = 0;
    this->bottom_id2_ = 0;
    this->whole_ = true;
  }
  this->top_idx_     = this->top_id_ * 2;
  this->top_idx2_    = this->top_id2_ * 2;
  this->bottom_idx_  = this->bottom_id_ * 2;
  this->bottom_idx2_ = this->bottom_id2_ * 2;

  CHECK_GE(this->top_id_,     0);
  CHECK_GE(this->top_id2_,    0);
  CHECK_GE(this->bottom_id_,  0);
  CHECK_GE(this->bottom_id2_, 0);

  bool is_disp_info = this->layer_param_.is_disp_info();
  if(is_disp_info) {
    LOG(INFO) << "top_id: "     << this->top_id_ 
        << ", top_idx: " << this->top_idx_;
    LOG(INFO) << "top_id2: "    << this->top_id2_ 
        << ", top_idx2: " << this->top_idx2_;
    LOG(INFO) << "bottom_id: "  << this->bottom_id_ 
        << ", bottom_idx: " << this->bottom_idx_;
    LOG(INFO) << "bottom_id2: " << this->bottom_id2_ 
        << ", bottom_idx2: " << this->bottom_idx2_;
    LOG(INFO) << "value: "      << this->value_;
    LOG(INFO) << "whole: "      << this->whole_;
    if(this->has_visual_path_) {
      LOG(INFO) << "visual_path: " << this->visual_path_;
    }
  }
}

// bottom[0]: coords -- have been scaled
// bottom[1]: aux_info -- origin height & width
template <typename Dtype>
void CoordsToBboxesMasksLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
  int num      = bottom[0]->num();
  int count    = bottom[0]->count();
  int channels = bottom[0]->channels();
  CHECK_LT(this->top_id_,     channels);
  CHECK_LT(this->top_id2_,    channels);
  CHECK_LT(this->bottom_id_,  channels);
  CHECK_LT(this->bottom_id2_, channels);
  CHECK_EQ(num,  bottom[1]->num());
  CHECK_EQ(channels, count / num);

  this->num_        = num;
  this->channels_   = channels;
  this->n_channels_ = 1;

  Dtype max_height      = Dtype(-1);
  Dtype max_width       = Dtype(-1);
  const Dtype* aux_info = bottom[1]->cpu_data();

  for(int item_id = 0; item_id < this->num_; item_id++) {
    // (img_ind, width, height, im_scale, flippable)
    const int offset = bottom[1]->offset(item_id);
    // const Dtype img_ind   = aux_info[offset + 0];
    const Dtype width     = aux_info[offset + 1];
    const Dtype height    = aux_info[offset + 2];
    const Dtype im_scale  = aux_info[offset + 3];
    // const Dtype flippable = aux_info[offset + 4];
    const Dtype r_width  = width * im_scale;
    const Dtype r_height = height * im_scale;
    max_width  = std::max(max_width,  r_width);
    max_height = std::max(max_height, r_height);
  }
  this->width_ = int(max_width);
  this->height_ = int(max_height);
  // Reshape  
  if(this->height_ <= 0 || this->width_ <= 0) {
    // You must know where to set `g_width` and `g_height`
    // Just for the initialization, like the deploy.prototxt
    //    in tools/camera_pose.cpp
    // The shape must keep the same as input layer
    const int g_width  = GlobalVars::g_width();
    const int g_height = GlobalVars::g_height();
    top[0]->Reshape(this->num_, 
        this->n_channels_, g_width, g_height);
  } else {
    top[0]->Reshape(this->num_, 
        this->n_channels_, this->height_, this->width_);
  }
  // 
  this->InitRand();
}

template <typename Dtype>
void CoordsToBboxesMasksLayer<Dtype>::InitRand() {
  const unsigned int rng_seed = caffe_rng_rand();
  rng_.reset(new Caffe::RNG(rng_seed));
}

template <typename Dtype>
int CoordsToBboxesMasksLayer<Dtype>::Rand(int n) {
  CHECK(rng_);
  CHECK_GT(n, 0);
  caffe::rng_t* rng =
      static_cast<caffe::rng_t*>(rng_->generator());
  return ((*rng)() % n);
}

// coords: has been rescaled by some layer, like data layer or ... 
template <typename Dtype>
void CoordsToBboxesMasksLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
  int mo = 0;
  int pco = 0;
  Dtype x1, y1, x2, y2;

  const Dtype Zero         = Dtype(0);
  const Dtype* part_coords = bottom[0]->cpu_data();
  Dtype* bbox_masks        = top[0]->mutable_cpu_data();
  caffe_set(top[0]->count(), Zero, bbox_masks);

  const Dtype perb_num = this->perb_num_ * 2 + 1;

  for(int n = 0; n < this->num_; n++) {
    pco = bottom[0]->offset(n);
    if(this->whole_) {
      x1 = part_coords[pco];
      x2 = part_coords[pco];
      y1 = part_coords[pco + 1];
      y2 = part_coords[pco + 1];
      for(int c = 0; c < this->channels_; c += 2) {
        x1 = std::min(x1, part_coords[pco + c + 0]);
        x2 = std::max(x2, part_coords[pco + c + 0]);
        y1 = std::min(y1, part_coords[pco + c + 1]);
        y2 = std::max(y2, part_coords[pco + c + 1]);
      }
    } else {
      const int random_value = this->Rand(2);
      if(random_value) {
        x1 = part_coords[pco + this->top_idx_ + 0];
        y1 = part_coords[pco + this->top_idx_ + 1];
        x2 = part_coords[pco + this->bottom_idx_ + 0];
        y2 = part_coords[pco + this->bottom_idx_ + 1];
      } else {
        x1 = part_coords[pco + this->top_idx2_ + 0];
        y1 = part_coords[pco + this->top_idx2_ + 1];
        x2 = part_coords[pco + this->bottom_idx2_ + 0];
        y2 = part_coords[pco + this->bottom_idx2_ + 1];
      }
    }
    // sometime we need to inorge some images
    if(x1 <= 0 && y1 <= 0 && x2 <= 0 && y2 <= 0) continue;
    if(x1 >= this->width_ && y1 >= this->height_
        && x2 >= this->width_ && y2 >= this->height_) {
      continue;
    }

    // pertube
    if(this->is_perturb_) {
      const int ind   = this->Rand(20);
      const Dtype pn  = Dtype(this->Rand(perb_num) - this->perb_num_);
      const Dtype pn2 = Dtype(this->Rand(perb_num) - this->perb_num_);
      const Dtype pn3 = Dtype(this->Rand(perb_num) - this->perb_num_);
      const Dtype pn4 = Dtype(this->Rand(perb_num) - this->perb_num_);
      switch(ind) {
        // single 
        case 1: x1 += pn;
          break;
        case 2: x1 += pn;
          break;
        case 3: y1 += pn;
          break;
        case 4: y1 += pn;
          break;
        case 5: x2 += pn;
          break;
        case 6: x2 += pn;
          break;
        case 7: y2 += pn;
          break;
        case 8: y2 += pn;
          break;
        // both && all
        case 11: x1 += pn;  y1 += pn;
          break;
        case 12: x1 += pn;  y1 += pn;
          break;
        case 13: x2 += pn;  y2 += pn;
          break;
        case 14: x2 += pn;  y2 += pn;
          break;
        case 15: x1 += pn;  y1 += pn;  x2 += pn2; y2 += pn2;
          break;
        case 16: x1 += pn;  y1 += pn;  x2 += pn2; y2 += pn2;
          break;
        case 17: x1 += pn;  y1 += pn2; x2 += pn3; y2 += pn4;
          break;
        case 18: x1 += pn;  y1 += pn2; x2 += pn3; y2 += pn4;
          break;
        default:
          break;
      }
      if(this->layer_param_.is_disp_info()) {
        LOG(INFO) << "ind: "     << ind << " " 
                  << "perb_n: "  << pn  << " "
                  << "perb_n2: " << pn2 << " "
                  << "perb_n3: " << pn3 << " "
                  << "perb_n4: " << pn4;
      }
    }

    // check bound
    if(x1 > x2) std::swap(x1, x2);
    if(y1 > y2) std::swap(y1, y2);
    if(x1 < Zero || x1 >= this->width_)  continue;
    if(x2 < Zero || x2 >= this->width_)  continue;
    if(y1 < Zero || y1 >= this->height_) continue;
    if(y2 < Zero || y2 >= this->height_) continue;
    // set the corresponding bbox/area to non-zero value
    for(int h = y1; h <= y2; h++) {
      for(int w = x1; w <= x2; w++) {
        mo = top[0]->offset(n, 0, h, w);
        bbox_masks[mo] = this->value_;
      }
    }
  }

  // visualization
  if(this->has_visual_path_) {
    const std::vector<std::string>& objidxs = 
        GlobalVars::objidxs();
    const std::vector<std::string>& imgidxs = 
        GlobalVars::imgidxs();
    CHECK_EQ(objidxs.size(), imgidxs.size());
    CHECK_EQ(objidxs.size(), this->num_);
    
    const Dtype* bbox_masks = top[0]->cpu_data();
    for(int n = 0; n < this->num_; n++) {
      cv::Mat img = cv::Mat::zeros(this->height_, this->width_, CV_8UC1);
      for(int h = 0; h < this->height_; h++) {
        for(int w = 0; w < this->width_; w++) {
          const int offset = top[0]->offset(n, 0, h, w);
          // (rows, cols) <---> (height, width)
          img.at<uchar>(h, w) = bbox_masks[offset];
        }
      }
      // 
      const std::string img_path = this->visual_path_ 
          + imgidxs[n] + "_" + objidxs[n] + this->img_ext_;

      const std::string dire = DireName(img_path);
      CreateDir(dire.c_str(), 0);
      
      cv::imwrite(img_path, img);

      // cv::imshow(img_path, img);
      // cv::waitKey(0);
      // cv::destroyAllWindows();
    }
  }
}

template <typename Dtype>
void CoordsToBboxesMasksLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, 
    const vector<Blob<Dtype>*>& bottom) 
{
  const Dtype Zero = Dtype(0);
  CHECK_EQ(propagate_down.size(), bottom.size());

  for (int i = 0; i < propagate_down.size(); ++i) {
    if (propagate_down[i]) { 
      // NOT_IMPLEMENTED; 
      caffe_set(bottom[i]->count(), Zero, 
          bottom[i]->mutable_cpu_diff());
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(CoordsToBboxesMasksLayer);
#endif

INSTANTIATE_CLASS(CoordsToBboxesMasksLayer);
REGISTER_LAYER_CLASS(CoordsToBboxesMasks);

}  // namespace caffe