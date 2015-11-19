#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/pose_estimation_layers.hpp"

namespace caffe {

template <typename Dtype>
void NormalizedPoseCoordsLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) 
{
  CHECK(this->layer_param_.has_norm_pose_coords_param());
  const NormPoseCoordsParameter norm_pose_coords_param = 
      this->layer_param_.norm_pose_coords_param();

  CHECK(norm_pose_coords_param.has_normalized_type());
  this->normalized_type_ = norm_pose_coords_param.normalized_type();
  CHECK_GE(this->normalized_type_, 0);
  CHECK_LE(this->normalized_type_, 2);

  this->has_statistic_file_ = false;
  if(this->normalized_type_ == 2) {
    CHECK(norm_pose_coords_param.has_statistic_file());
    this->statistic_file_ = norm_pose_coords_param.statistic_file();
    LOG(INFO) << "statistic_file: " << this->statistic_file_;
    this->has_statistic_file_ = true;
    // file handler
    std::ifstream in_file(this->statistic_file_.c_str());
    CHECK(in_file);
    // variables
    int lnum;
    float coord_ave;
    float coord_std;
    // read from file
    in_file >> lnum;
    LOG(INFO) << "lable_num (from statistic_file): " << lnum;
    for(int tn = 0; tn < lnum; tn++) {
      in_file >> coord_ave >> coord_std;
      this->coord_aves_.push_back(coord_ave);
      this->coord_stds_.push_back(coord_std);
      LOG(INFO) << "idx: " << tn << " -- ave: " << coord_ave  << ", std: " << coord_std;
    }

    in_file.close();
    std::cout << std::endl << std::endl;
  }
}

template <typename Dtype>
void NormalizedPoseCoordsLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) 
{
  this->num_ = bottom[0]->num();
  this->channels_ = bottom[0]->channels();
  this->height_ = bottom[0]->height();
  this->width_ = bottom[0]->width();
  CHECK_EQ(this->channels_, bottom[0]->count() / this->num_);

  if(this->has_statistic_file_) {
    CHECK_EQ(this->channels_, this->coord_aves_.size());
    CHECK_EQ(this->channels_, this->coord_stds_.size());
  }

  if(this->normalized_type_ != 0) {
    // use aux info, (img_ind, width, height, im_scale, flippable)
    CHECK_EQ(bottom.size(), 2);
    CHECK_EQ(bottom[1]->num(), bottom[0]->num());
    CHECK_EQ(bottom[1]->channels(), bottom[1]->count() / bottom[1]->num());
    CHECK_EQ(bottom[1]->channels(), 5);
  }

  // reshape
  top[0]->Reshape(
      this->num_,
      this->channels_,
      this->height_,
      this->width_
  );
}

template <typename Dtype>
void NormalizedPoseCoordsLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) 
{
  const Dtype* bottom_coords_ptr = bottom[0]->cpu_data();
  Dtype* top_coords_ptr = top[0]->mutable_cpu_data();

  // do nothing, just copy
  if(this->normalized_type_ == 0) {
    caffe_copy(
        bottom[0]->count(), 
        bottom_coords_ptr, 
        top_coords_ptr
    );
  // deeppose
  } else if(this->normalized_type_ == 1) {
    int coords_offset = 0;
    int aux_info_offset = 0;
    Dtype x, y, height, width, im_scale, half_height, half_width;
    // get aux info pointer
    const Dtype* aux_info_ptr = bottom[1]->cpu_data();

    for(int n = 0; n < this->num_; n++) {
      // get offset
      coords_offset = bottom[0]->offset(n);
      aux_info_offset = bottom[1]->offset(n);
      // get aux info (img_ind, width, height, im_scale, flippable)
      im_scale = aux_info_ptr[aux_info_offset + 3];
      // multiply the im_scale
      height = aux_info_ptr[aux_info_offset + 1] * im_scale;
      width = aux_info_ptr[aux_info_offset + 2] * im_scale;
      half_height = height / Dtype(2);
      half_width = width / Dtype(2);

      // process each pos/loc
      for(int c = 0; c < this->channels_; c +=2) {
        x = bottom_coords_ptr[coords_offset + c];
        y = bottom_coords_ptr[coords_offset + c + 1];
        // normalized
        x = (x - half_width) / width;
        y = (y - half_height) / height;

        // set
        top_coords_ptr[coords_offset + c] = x;
        top_coords_ptr[coords_offset + c + 1] = y;
      }
    }
  // fashion parsing
  } else if(this->normalized_type_ == 2) {
    int coords_offset = 0;
    int aux_info_offset = 0;
    Dtype coor, im_scale, ave, std;
    // get aux info pointer
    const Dtype* aux_info_ptr = bottom[1]->cpu_data();

    // process each pos/loc
    for(int n = 0; n < this->num_; n++) {
      // get offset
      coords_offset = bottom[0]->offset(n);
      aux_info_offset = bottom[1]->offset(n);
      // get aux info (img_ind, width, height, im_scale, flippable)
      im_scale = aux_info_ptr[aux_info_offset + 3];
      
      // process each pos/loc
      for(int c = 0; c < this->channels_; c++) {
        coor = bottom_coords_ptr[coords_offset + c];
        ave = this->coord_aves_[c] * im_scale;
        std = this->coord_stds_[c] * im_scale;
        // normalized
        coor = (coor - ave) / std;

        // set
        top_coords_ptr[coords_offset + c] = coor;
      }
    }
  }
}

template <typename Dtype>
void NormalizedPoseCoordsLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
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
STUB_GPU(NormalizedPoseCoordsLayer);
#endif

INSTANTIATE_CLASS(NormalizedPoseCoordsLayer);
REGISTER_LAYER_CLASS(NormalizedPoseCoords);

}  // namespace caffe