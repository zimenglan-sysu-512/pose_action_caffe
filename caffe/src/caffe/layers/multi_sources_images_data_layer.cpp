#include <opencv2/core/core.hpp>

#include <fstream>   // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "boost/algorithm/string.hpp"
#include "google/protobuf/text_format.h"

#include "caffe/pose_estimation_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/common.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/data_layers.hpp"
#include "caffe/util/util_img.hpp"
#include "caffe/util/pose_tool.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/global_variables.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/convert_img_blob.hpp"

#define __MULTI_SOURCES_IMAGES_DATA_LAYER_VISUAL__
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

namespace caffe {

template <typename Dtype>
MultiSourcesImagesDataLayer<Dtype>::~MultiSourcesImagesDataLayer<Dtype>() {
  this->JoinPrefetchThread();
}

template <typename Dtype>
void MultiSourcesImagesDataLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
  BaseDataLayer<Dtype>::LayerSetUp(bottom, top);

  // Now, start the prefetch thread. Before calling prefetch, we make two
  // cpu_data calls so that the prefetch thread does not accidentally make
  // simultaneous cudaMalloc calls when the main thread is running. In some
  // GPUs this seems to cause failures if we do not so.
  LOG(INFO) << "data/images...";
  this->prefetch_data_.mutable_cpu_data();
  this->transformed_data_.mutable_cpu_data();

  if(top.size() > 1) {
    this->output_labels_ = true;

    LOG(INFO)               << "labels/coordinates, aux info...";
    CHECK_EQ(top.size(), 3) << "labels/coordinates, aux info..."; 

    this->prefetch_label_.mutable_cpu_data();
    this->aux_info_.mutable_cpu_data();
  }

  DLOG(INFO) << "Initializing prefetch";
  this->CreatePrefetchThread();
  DLOG(INFO) << "Prefetch initialized.";
}

// data, labels, im_scales
template <typename Dtype>
void MultiSourcesImagesDataLayer<Dtype>::DataLayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
  CHECK(this->layer_param_.has_multi_sources_images_data_param());
  const MultiSourcesImagesDataParameter msidp = 
      this->layer_param_.multi_sources_images_data_param();

  CHECK(msidp.source_size()     == 1) 
        << "Only one input label file availble.";
  CHECK(msidp.root_folder_size() > 0);
  CHECK(msidp.im_exts_size()     > 0);
  CHECK(msidp.is_colors_size()   > 0);

  this->n_sources_ = msidp.root_folder_size();
  CHECK_EQ(this->n_sources_, msidp.root_folder_size());
  CHECK_EQ(this->n_sources_, msidp.im_exts_size());
  CHECK_EQ(this->n_sources_, msidp.is_colors_size());

  CHECK(msidp.has_shuffle());
  CHECK(msidp.has_label_num());
  CHECK(msidp.has_batch_size());
  CHECK(msidp.has_is_scale_image());
  CHECK(msidp.has_parts_orders_path());

  this->is_scale_image_       = msidp.is_scale_image();
  this->label_num_            = msidp.label_num();
  this->receptive_field_size_ = 0;

  if(msidp.has_receptive_field_size()) {
    this->receptive_field_size_ = msidp.receptive_field_size();
    CHECK(this->receptive_field_size_ > 0);
  }

  // scale and angle
  if(!this->scales_.empty()) {
    this->scales_.clear();
  }
  if(msidp.has_scale_string()) {
    std::string scale_string = msidp.scale_string();
    boost::trim(scale_string);
    std::vector<std::string> scale_info;
    boost::split(scale_info, scale_string, boost::is_any_of(","));
    for(int j = 0; j < scale_info.size(); j++) {
      this->scales_.push_back(std::atof(scale_info[j].c_str()));
    }
  } else {
    this->scales_.push_back(1.0);
  }
  CHECK_GE(this->scales_.size(), 1);
  for(int j = 0; j < this->scales_.size(); j++) {
    LOG(INFO) << "ind: " << j << " -- scales: " << this->scales_[j];
  }

  this->has_angle_   = false;
  if(msidp.has_angle_max()) {
    this->has_angle_ = true;
    this->angle_max_ = msidp.angle_max();
    CHECK_GT(this->angle_max_, 0);
  }
  LOG(INFO) << "has angle: " << this->has_angle_;
  if(this->has_angle_) {
    LOG(INFO) << "angle_max: " << this->angle_max_;
  }

  // need rotate
  if(!this->need_rotate_ims_.empty()) {
    this->need_rotate_ims_.clear();
  }
  if(msidp.need_rotates_size() > 0) {
    CHECK(msidp.need_rotates_size() == 1 || 
          msidp.need_rotates_size() == this->n_sources_);
    for(int j = 0; j < msidp.need_rotates_size(); j++) {
      const bool flag = msidp.need_rotates(j);
      this->need_rotate_ims_.push_back(flag);
    }
    if(msidp.need_rotates_size() == 1) {
      const bool flag = this->need_rotate_ims_[0];
      for(int j = 1; j < this->n_sources_ ; j++) {
        this->need_rotate_ims_.push_back(flag);
      }
    }
    
  } else {
    for(int j = 0; j < this->n_sources_ ; j++) {
      this->need_rotate_ims_.push_back(true); // default: depend on has_angle_
    }
  }
  CHECK_EQ(this->need_rotate_ims_.size(), this->n_sources_);
  for(int j = 0; j < this->need_rotate_ims_.size(); j++) {
    LOG(INFO) << "ind: " << j << " need rotate im: " 
              << this->need_rotate_ims_[j];
  }

  this->rotate_prob_num_ = 0;
  if(this->has_angle_) {
    CHECK(msidp.has_rotate_prob_num());
    this->rotate_prob_num_ = msidp.rotate_prob_num();
    CHECK_GT(this->rotate_prob_num_, 1);
  }
  LOG(INFO) << "rotate_prob_num: " << this->rotate_prob_num_;


  // need translate
  bool need_translate_flag = false;
  if(!this->need_translate_ims_.empty()) {
    this->need_translate_ims_.clear();
  }
  if(msidp.need_translates_size() > 0) {
    CHECK(msidp.need_translates_size() == 1 || 
          msidp.need_translates_size() == this->n_sources_);
    for(int j = 0; j < msidp.need_translates_size(); j++) {
      const bool flag = msidp.need_translates(j);
      this->need_translate_ims_.push_back(flag);
      if(flag) {
        need_translate_flag = true;
      }
    }
    if(msidp.need_translates_size() == 1) {
      const bool flag = this->need_translate_ims_[0];
      for(int j = 1; j < this->n_sources_ ; j++) {
        this->need_translate_ims_.push_back(flag);
      }
    }
    
  } else {
    for(int j = 0; j < this->n_sources_ ; j++) {
      this->need_translate_ims_.push_back(false);
    }
  }
  CHECK_EQ(this->need_translate_ims_.size(), this->n_sources_);
  for(int j = 0; j < this->need_translate_ims_.size(); j++) {
    LOG(INFO) << "ind: " << j << " translate step: " 
              << this->need_translate_ims_[j];
  }

  this->translate_num_ = 0;
  this->translate_prob_num_ = 0;

  if(need_translate_flag) {
    CHECK(msidp.has_translate_num());
    this->translate_num_ = msidp.translate_num();
    CHECK_GT(this->translate_num_, 1);

    CHECK(msidp.has_translate_prob_num());
    this->translate_prob_num_ = msidp.translate_prob_num();
    CHECK_GT(this->translate_prob_num_, 1);
  }
  LOG(INFO) << "translate_num: "      << this->translate_num_;
  LOG(INFO) << "translate_prob_num: " << this->translate_prob_num_;

  // scale image
  if(this->is_scale_image_) {
    CHECK(msidp.has_max_size());
    CHECK(msidp.min_size_size() > 0);
    CHECK(msidp.has_always_max_size());

    this->max_size_        = msidp.max_size();
    this->always_max_size_ = msidp.always_max_size();
    CHECK_GT(this->max_size_, 0);
  
    for(int ms = 0; ms < msidp.min_size_size(); ms++) {
      const int min_size = std::min(msidp.min_size(ms), 
                                    this->max_size_);
      CHECK_GT(min_size, 0);
      this->min_sizes_.push_back(min_size);
    }

    CHECK_GE(this->min_sizes_.size(), 1);
  }
  CHECK_EQ(this->label_num_ % 2, 0);

  // check if we want to use mean_value
  const TransformationParameter transform_param = 
      this->layer_param_.transform_param();
  CHECK(transform_param.has_mean_file() == false) 
      << "Cannot specify mean_file and mean_value at the same time"
      << "\n*** Of cause, here, "
      << "we restrict that we can't use mean_file. ***\n";

  this->has_mean_values_ = false;
  const int mean_value_size = transform_param.mean_value_size();
  if (mean_value_size > 0) {
    CHECK((mean_value_size == 1 || mean_value_size == 3));
    // opencv: 0: blue; 1: green; 2: red
    for (int c = 0; c < mean_value_size; ++c) {
      const float mean_value = transform_param.mean_value(c);
      this->mean_values_.push_back(mean_value);
    }
    if (this->mean_values_.size() == 1) {
      // Replicate the mean_value for simplicity
      for (int c = 1; c < 3; ++c) {
        mean_values_.push_back(mean_values_[0]);
      }
    }
    LOG(INFO) << "mean_values: (" 
              << this->mean_values_[0] <<  ", " 
              << this->mean_values_[1] <<  ", " 
              << this->mean_values_[2] <<  ")"; 
    this->has_mean_values_ = true;
  }

  // flippable
  this->parts_orders_path_ = msidp.parts_orders_path();
  if(this->origin_parts_orders_.size()) {
    this->origin_parts_orders_.clear();
  }
  if(this->flippable_parts_orders_.size()) {
    this->flippable_parts_orders_.clear();
  }
  CHECK(!this->origin_parts_orders_.size() && 
        !this->flippable_parts_orders_.size());
  get_parts_orders(this->parts_orders_path_, 
                   this->origin_parts_orders_, 
                   this->flippable_parts_orders_);
  CHECK(this->origin_parts_orders_.size() && 
        this->flippable_parts_orders_.size());
  CHECK_EQ(this->origin_parts_orders_.size(), 
           this->flippable_parts_orders_.size());
  CHECK_EQ(this->origin_parts_orders_.size() * 2,
           this->label_num_);
  this->key_points_num_ = this->origin_parts_orders_.size();

  // clear
  if(!this->is_colors_.empty()) {
    this->is_colors_.clear();
  }
  if(!this->im_exts_.empty()) {
    this->im_exts_.clear();
  }
  if(!this->sources_.empty()) {
    this->sources_.clear();
  }
  if(!this->root_folders_.empty()) {
    this->root_folders_.clear();
  }

  // Read the file with imgidxs and labels
  for(int s = 0; s < msidp.source_size(); s++) {
    const std::string source  = msidp.source(s);
    LOG(INFO) << "\n\n" 
              << "s: " << s << " " 
              << "source: " << source << " "
              << "\n\n";
    this->sources_.push_back(source);
  }

  LOG(INFO) << "\n\nn_sources: " << this->n_sources_ << "\n\n";
  for(int s = 0; s < this->n_sources_; s++) {
    const std::string im_ext      = msidp.im_exts(s);
    const bool is_color           = msidp.is_colors(s);
    const std::string root_folder = msidp.root_folder(s);

    this->im_exts_.push_back(im_ext);
    
    this->is_colors_.push_back(is_color);
    this->root_folders_.push_back(root_folder);

    LOG(INFO) << "\n\n" 
              << "s: " << s << " " 
              << "im_ext: " << im_ext << " "
              << "is_color: " << is_color << " "
              << "root_folder: " << root_folder
              << "\n\n";
  }

  for(int s = 0; s < msidp.source_size(); s++) {
    const std::string source      = this->sources_[s];
    const std::string root_folder = this->root_folders_[s];

    LOG(INFO) << "Opening file " << source;
    std::ifstream infile(source.c_str());
    CHECK(infile);

    float pos;
    string objidx;
    string imgidx;
    string line_info;
    while (getline(infile, line_info)) {
      std::vector<std::string> label_info;
      // imgidx x1 y1 x2 y2 ... xn yn
      boost::trim(line_info);
      boost::split(label_info, line_info, boost::is_any_of(" "));

      CHECK_GE(label_info.size(), 2);
      if(label_info.size() == 2) {
        CHECK(this->output_labels_ == false);
      } else {
        CHECK_EQ(label_info.size(), this->label_num_ + 2);
      }
      // imgidx & objidx 
      imgidx = label_info[0];
      objidx = label_info[1];
      // labels
      std::vector<float> labels;  // x1, y1, x2, y2, ...
      if(this->output_labels_) {
        if(this->output_labels_) {
          for(int ln = 2; ln <= this->label_num_ + 1; ln++) {
            pos = std::atof(label_info[ln].c_str());
            labels.push_back(pos);
          }
        }  
      }
      if(this->layer_param_.is_disp_info()) {
        std::cout << imgidx << " " << objidx << " ";
        for(int j2 = 0; j2 < labels.size(); j2++) {
          std::cout << labels[j2] << " ";
        }
        std::cout << labels.size() << std::endl;
      }
      // record (source_idx, (imgidx, (objidx, labels)))
      lines_.push_back(std::make_pair(s, std::make_pair(imgidx, 
                          std::make_pair(objidx, labels))));
    }
  }

  // randomly shuffle data
  this->shuffle_ = msidp.shuffle();
  if (this->shuffle_) {
    LOG(INFO) << "Shuffling data";
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    // shuffle
    ShuffleImages();
  }
  LOG(INFO) << "\n\nA total of " << lines_.size() << " images.\n\n";

  // Check if we would need to randomly skip a few data points
  lines_id_ = 0;
  if (msidp.rand_skip()) {
    unsigned int skip = caffe_rng_rand() % msidp.rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    CHECK_GT(lines_.size(), skip)  << "Not enough points to skip";
    lines_id_ = skip;
  }

  // crop size & batch size
  this->batch_size_   = msidp.batch_size();
  this->crop_size_    = transform_param.crop_size();
  CHECK_EQ(this->crop_size_, 0) << "does not need cropping...";

  // Read an image, and use it to initialize the top blob.
  const int source_idx = lines_[lines_id_].first;
  const std::string image_path = this->root_folders_[source_idx] 
                               + lines_[lines_id_].second.first 
                               + this->im_exts_[source_idx];
  // image 
  cv::Mat im          = ImageRead(image_path, 
                                  this->is_colors_[source_idx]);
  const int max_size  = 100;
  const int width     = im.cols > max_size ? max_size : im.cols;
  const int height    = im.rows > max_size ? max_size : im.rows;
  // const int channels  = im.channels();
  
  // total channels
  if(!this->channels_inds_.empty()) {
    this->channels_inds_.clear();
  }
  int t_channels = 0;
  LOG(INFO) << "n_sources: " << this->n_sources_;
  for(int j2 = 0; j2 < this->n_sources_; j2++) {
    LOG(INFO) << "t_channels: " << t_channels;
    this->channels_inds_.push_back(t_channels);
    t_channels = this->is_colors_[j2] ? (t_channels + 3) 
                                      : (t_channels + 1);
  }
  LOG(INFO) << "t_channels: " << t_channels;
  this->channels_inds_.push_back(t_channels);

  // data/images
  top[0]->Reshape(this->batch_size_, t_channels, height, width);
  this->prefetch_data_.Reshape(this->batch_size_, t_channels, 
                               height, width);
  this->transformed_data_.Reshape(1, t_channels, height, width);
  
  LOG(INFO) << "\n\noutput data size: " 
            << top[0]->num()      << ","
            << top[0]->channels() << "," 
            << top[0]->height()   << ","
            << top[0]->width()    << "\n\n";

  // labels & aux info
  if(top.size() > 1) {
    CHECK_EQ(top.size(), 3) 
        << "Reshape: data/images, labels/coordinates, aux info...";
    // label
    top[1]->Reshape(this->batch_size_, this->label_num_, 1, 1);
    this->prefetch_label_.Reshape(this->batch_size_, 
                                  this->label_num_, 1, 1);
    LOG(INFO);
    LOG(INFO) << this->label_num_ << ", " 
              << this->batch_size_;
    LOG(INFO) << "output prefetch_label_ size: " 
              << top[1]->count();
    LOG(INFO) << "output prefetch_label_ shape_string: " 
              << top[1]->shape_string();
    LOG(INFO);
    
    // (img_ind, width, height, im_scale, flippable)
    top[2]->Reshape(this->batch_size_, 5, 1, 1);
    this->aux_info_.Reshape(this->batch_size_, 5, 1, 1);

    LOG(INFO) << "output aux_info_ size: " << top[2]->count();
    LOG(INFO) << "output aux_info_ shape_string: " 
              << top[2]->shape_string();
    LOG(INFO);

    // first initialize for create heat maps from coordinates
    // see `heat_maps_from_coords_layer.cpp`
    Dtype* aux_info = top[2]->mutable_cpu_data();
    for(int bs = 0; bs < this->batch_size_; bs++) {
      const int  ai_offset      = top[2]->offset(bs);
        aux_info[ai_offset + 0] = Dtype(bs);
        aux_info[ai_offset + 1] = Dtype(width);
        aux_info[ai_offset + 2] = Dtype(height);
        aux_info[ai_offset + 3] = Dtype(1.);
        aux_info[ai_offset + 4] = Dtype(0);
    }
  }
}

template <typename Dtype>
void MultiSourcesImagesDataLayer<Dtype>::PerturbedCoordsBias() {
}

template <typename Dtype>
float MultiSourcesImagesDataLayer<Dtype>::Uniform(
    const float min, const float max) 
{
  int diff = int(max - min);
  if(diff < 0) diff = 0 - diff;
  CHECK_GT(diff, 0);
  // LOG(INFO) << "uniform 0 -- diff: " << diff;
  int random = this->data_transformer_->Rand(diff * 10 + 1);
  float r    = random * 0.1;
  // LOG(INFO) << "uniform 1 -- random: " << random << " r: " << r;
  return min + r;
}

template <typename Dtype>
void MultiSourcesImagesDataLayer<Dtype>::ShuffleImages() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

// This function is used to create a thread that prefetches the data.
template <typename Dtype>
void MultiSourcesImagesDataLayer<Dtype>::InternalThreadEntry() {
  // timer
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  CPUTimer timer;

  // initialize
  CHECK(this->prefetch_data_.count());
  CHECK(this->transformed_data_.count());
  if(this->output_labels_) {
    CHECK(this->aux_info_.count());
    CHECK(this->prefetch_label_.count());
  }

  // Dtype* prefetch_data = this->prefetch_data_.mutable_cpu_data();
  Dtype* aux_info       = NULL;
  Dtype* prefetch_label = NULL;
  if(this->output_labels_) {
    aux_info       = this->aux_info_.mutable_cpu_data();
    prefetch_label = this->prefetch_label_.mutable_cpu_data();
  }

  // variables
  std::vector<int> widths;
  std::vector<int> heights;
  std::vector<float> im_scales;
  std::vector<std::vector<float> > labels;

  // clear before setting values
  if(this->objidxs_.size() > 0) {
    this->objidxs_.clear();
  }
  CHECK_EQ(this->objidxs_.size(), 0);
  if(this->imgidxs_.size() > 0) {
    this->imgidxs_.clear();
  }
  CHECK_EQ(this->imgidxs_.size(), 0);
  if(this->images_paths_.size() > 0) {
    this->images_paths_.clear();
  }
  CHECK_EQ(this->images_paths_.size(), 0);

  if(this->layer_param_.is_disp_info()) {
    LOG(INFO) << "\nStart calculating size and scale -- 0...\n";
  }

  // get size and scale
  timer.Start();
  const int lines_size = lines_.size();
  for (int item_id = 0; item_id < this->batch_size_; ++item_id) {
    CHECK_GT(lines_size, lines_id_);

    const int source_idx         = lines_[lines_id_].first;
    const std::string imgidx     = lines_[lines_id_].second.first;
    const std::string objidx     = lines_[lines_id_].second.second.first;
    const std::string image_path = this->root_folders_[source_idx] + 
                                   imgidx + this->im_exts_[source_idx];
    this->objidxs_.push_back(objidx);
    this->imgidxs_.push_back(imgidx);
    this->images_paths_.push_back(image_path);

    cv::Mat im = ImageRead(image_path, this->is_colors_[source_idx]);
    CHECK(im.data) << "Could not load " << image_path;
    const int width  = im.cols;
    const int height = im.rows;
    widths.push_back(width);
    heights.push_back(height);

    // labels
    if(this->output_labels_) {
      const std::vector<float>& label = 
          lines_[lines_id_].second.second.second;
      CHECK_EQ(label.size(), this->label_num_);
      labels.push_back(label);
    }
    
    // go to the next iter
    lines_id_++;
    if (lines_id_ >= lines_size) {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      lines_id_ = 0;
      if (this->shuffle_) {
        ShuffleImages();
      }
    }
  }
  CHECK_EQ(widths.size(),              this->batch_size_);
  CHECK_EQ(heights.size(),             this->batch_size_);
  CHECK_EQ(this->objidxs_.size(),      this->batch_size_);
  CHECK_EQ(this->imgidxs_.size(),      this->batch_size_);
  CHECK_EQ(this->images_paths_.size(), this->batch_size_);
  if(this->output_labels_) {
    CHECK_EQ(labels.size(), this->batch_size_);
  }
  read_time += timer.MicroSeconds();

  if(this->layer_param_.is_disp_info()) {
    LOG(INFO) << "\nStart calculating size and scale -- 1...\n";
  }

  // image scale
  timer.Start();
  if(this->is_scale_image_) {
    const int n_scales   = this->scales_.size();
    int scale_ind = 0;
    if(n_scales > 1) {
      scale_ind  = this->data_transformer_->Rand(n_scales);
    }
    const float scale    = this->scales_[scale_ind];
    float max_size       = float(this->max_size_) * scale;
    // LOG(INFO) << "scale_ind: " << scale_ind << " "
    //           << "scale: " << scale << " max_size: " << max_size;
    
    for (int item_id = 0; item_id < this->batch_size_; ++item_id) {
      float im_min_size = std::min(widths[item_id], heights[item_id]);
      float im_max_size = std::max(widths[item_id], heights[item_id]);

      float im_scale;
      if(this->always_max_size_) {
        im_scale = float(max_size) / float(im_max_size + 0.0);
      } else {
        const int s   = this->min_sizes_.size();
        int ind = 0;
        if(s > 1) {
          ind = this->data_transformer_->Rand(s);
        }
        // LOG(INFO) << "min_size ind: " << ind;
        const int min_size = this->min_sizes_[ind];
        
        max_size = std::max(max_size,float(min_size));

        im_scale = float(std::max(min_size, int(im_min_size))) 
                 / float(std::min(min_size, int(im_min_size)) + 0.);
        if(im_scale * im_max_size > max_size) {
          im_scale = float(max_size) / float(im_max_size + 0.0);
        }
      }

      // scale factor
      im_scales.push_back(im_scale);
    }
  } else {
    for (int item_id = 0; item_id < this->batch_size_; ++item_id) {
      im_scales.push_back(1.0f);
    }
  }
  CHECK_EQ(im_scales.size(), this->batch_size_);
  
  // find the max height and max width along all the images alone
  int max_height = -1;
  int max_width  = -1;
  for (int item_id = 0; item_id < this->batch_size_; ++item_id) {
    max_width  = std::max(max_width,  int(widths[item_id] * 
                                          im_scales[item_id]));
    max_height = std::max(max_height, int(heights[item_id] * 
                                          im_scales[item_id]));
  }

  if(this->layer_param_.is_disp_info()) {
    LOG(INFO) << "\nStart reshape top blobs...\n";
  }

  // reshape & reset
  CHECK_EQ(this->n_sources_ + 1, this->channels_inds_.size());
  const int t_channels = this->channels_inds_[this->n_sources_];
  this->prefetch_data_.Reshape(this->batch_size_, t_channels, 
                               max_height, max_width);
  caffe_set(this->prefetch_data_.count(), 
            Dtype(0), 
            this->prefetch_data_.mutable_cpu_data());
  this->transformed_data_.Reshape(1, t_channels, 
                                  max_height, max_width);
  caffe_set(this->transformed_data_.count(), 
            Dtype(0), 
            this->transformed_data_.mutable_cpu_data());
  read_time += timer.MicroSeconds();

  // transform param
  const TransformationParameter transform_param = 
      this->layer_param_.transform_param();
      
  if(this->layer_param_.is_disp_info()) {
    LOG(INFO) << "Start fetching images & labels...";
  }

  // images & labels (rescale)
  for (int item_id = 0; item_id < this->batch_size_; ++item_id) {
    if(this->layer_param_.is_disp_info()) {
      LOG(INFO) << "Start fetching images & labels -- " << item_id << "...";
    }
    
    timer.Start();
    // mirror -- horizontal
    // LOG(INFO) << "item_id: " << item_id;
    const bool do_mirror = transform_param.mirror() && 
                           this->data_transformer_->Rand(2);
    // LOG(INFO) << "mirror: " << do_mirror;
    
    // rotate and translate
    // LOG(INFO) << "rotate 0 -- has_angle: " << this->has_angle_;
    cv::Mat M( 2, 3, CV_32FC1);
    cv::Mat M2(2, 3, CV_64FC1);

    const int translate_prob_num = this->translate_prob_num_ <= 1 ? 1 : 
                                   this->data_transformer_->Rand(this->translate_prob_num_);
    const bool do_translate = translate_prob_num == 0;

    const int rotate_prob_num = this->rotate_prob_num_ <= 1 ? 1 : 
                           this->data_transformer_->Rand(this->rotate_prob_num_);
    const bool do_rotate = this->has_angle_ && (rotate_prob_num == 0);


    // LOG(INFO) << "translate 1 -- do_translate: " << do_translate  << " "
    //           << "translate_prob_num: " << translate_prob_num;
    // LOG(INFO) << "rotate 1 -- do_rotate: "       << do_rotate     << " "
    //           << "rotate_prob_num: " << rotate_prob_num;
    
    bool first_rotate    = true;
    bool first_translate = true;
    
    // images from multi-sources
    for(int j2 = 0; j2 < this->n_sources_; j2++) {
      const std::string image_path = this->root_folders_[j2] +
                                     this->imgidxs_[item_id] + 
                                     this->im_exts_[j2];
      // cv::Mat im = ReadImageToCVMat(image_path, height, width, 
      //                               this->is_colors_[j2]);
      cv::Mat im;                                     
      bool flag = ResizeImage(image_path, im, widths[item_id], 
                              heights[item_id], im_scales[item_id], 
                              this->is_colors_[j2]);
      if(!flag) {
        if(this->layer_param_.is_disp_info()) {
          LOG(INFO) << "\n\nimage_path does not exist or invalid: " 
                    << image_path << "\n\n"; 
        }
        continue;
      }

      if(do_translate && first_translate) {
        const int tl1 = this->translate_num_;
        const int tl2 = tl1 * 2 + 1;
        int tl_dx     = this->data_transformer_->Rand(tl2);
        int tl_dy     = this->data_transformer_->Rand(tl2);
        tl_dx        -= tl1;
        tl_dy        -= tl1;
        // LOG(INFO) << "ind: " << j2 << " translate -- dx: " 
        //           << tl_dx << " dy: " << tl_dy;

        M2.at<double>(0, 0) = double(1);
        M2.at<double>(0, 1) = double(0);
        M2.at<double>(0, 2) = double(tl_dx);
        M2.at<double>(1, 0) = double(0);
        M2.at<double>(1, 1) = double(1);
        M2.at<double>(1, 2) = double(tl_dy);

        first_translate = false;
      }
      if(do_translate) {
        CHECK_EQ(first_translate, false);
      }

      if(do_rotate && first_rotate) {
        double s    = 1;
        cv::Point c = cv::Point(im.cols / 2, im.rows / 2);
        float angle = this->Uniform(-this->angle_max_, this->angle_max_);
        // Get the rotation matrix with the specifications above
        M = cv::getRotationMatrix2D(c, angle, s);
        first_rotate = false;
      }
      if(do_rotate) {
        CHECK_EQ(first_rotate, false);
      }
  
      // ####################################################### 
      // need to check <width, height> for all sources???
      // here do not check!!!
      // ToDo            
      // #######################################################    
      CHECK(im.data) << "Could not load " << image_path;

      // translate -- allow different source image to translate different (dx, dy)
      if(do_translate && this->need_translate_ims_[j2]) {
        // Translate 
        // LOG(INFO) << "ind2: " << j2 << " translate";
        cv::warpAffine(im, im, M2, im.size());
      }

      if(do_mirror) {
        // >0: horizontal; <0: horizontal&vertical; =0: vertical
        const int flipCode = 1;
        cv::flip(im, im, flipCode);
      }

      // rotate-- different source image has the same rotation
      if(do_rotate && this->need_rotate_ims_[j2]) {
        // Rotate the warped image
        cv::warpAffine(im, im, M, im.size());
      }

      // 把图片拷贝到放在blob的右上角
      int sc = this->channels_inds_[j2];
      if(this->has_mean_values_) {
        if(this->is_colors_[j2]) {  // color
          ImageDataToBlob(&(this->prefetch_data_), item_id, 
                          sc, im, this->mean_values_);
        } else {  // gray
          GrayImageDataToBlob(&(this->prefetch_data_), item_id, 
                              sc, im, this->mean_values_);
        }
        
      } else {  
        if(this->is_colors_[j2]) {  // color
          ImageDataToBlob(&(this->prefetch_data_), item_id, sc, im);
        } else {  // gray
          GrayImageDataToBlob(&(this->prefetch_data_), item_id, 
                              sc, im);
        }
      }
    }                  

    if(this->layer_param_.is_disp_info()) {
      LOG(INFO) << "Start fetching images -- mid...";
    }

    // timer
    read_time += timer.MicroSeconds();

    // labels && aux info
    if(this->output_labels_) {
      timer.Start();

      // labels
      const std::vector<float>& label = labels[item_id];
      CHECK_EQ(label.size(), this->label_num_);
      CHECK_EQ(label.size(), this->key_points_num_ * 2);

      const int label_offset = this->prefetch_label_.offset(item_id);
      for(int kpn = 0; kpn < this->key_points_num_; kpn++) {
        const int id = do_mirror ? this->flippable_parts_orders_[kpn] 
                                 : this->origin_parts_orders_[kpn];
        const int idx = id * 2;
        // x -- width
        float x =  label[idx + 0];
        float w = widths[item_id];
        // y -- height
        float y =  label[idx + 1];

        x = x * im_scales[item_id];
        w = w * im_scales[item_id];
        y = y * im_scales[item_id];
        
        // translate 
        if(do_translate && this->need_translate_ims_[0]) {
          const float x2 = M2.at<double>(0, 0) * x +
                           M2.at<double>(0, 1) * y + 
                           M2.at<double>(0, 2);
          const float y2 = M2.at<double>(1, 0) * x + 
                           M2.at<double>(1, 1) * y + 
                           M2.at<double>(1, 2);
          x = x2;
          y = y2;
        }

        // flip
        x = (do_mirror ? w - x - 1 : x);

        // rotate label
        if(do_rotate && this->need_rotate_ims_[0]) {
          const float x2 = M.at<double>(0, 0) * x +
                           M.at<double>(0, 1) * y + 
                           M.at<double>(0, 2);
          const float y2 = M.at<double>(1, 0) * x + 
                           M.at<double>(1, 1) * y + 
                           M.at<double>(1, 2);
          x = x2;
          y = y2;
        }
        
        // set
        const int ln = kpn * 2;
        prefetch_label[label_offset + ln + 0] = Dtype(x);
        prefetch_label[label_offset + ln + 1] = Dtype(y);
      }

      // aux info
      const int ai_offset     = this->aux_info_.offset(item_id);
      aux_info[ai_offset + 0] = Dtype(item_id);
      aux_info[ai_offset + 1] = Dtype(widths[item_id]);
      aux_info[ai_offset + 2] = Dtype(heights[item_id]);
      aux_info[ai_offset + 3] = Dtype(im_scales[item_id]);  
      aux_info[ai_offset + 4] = do_mirror ? Dtype(1) : Dtype(0);
    }

    // timer
    read_time += timer.MicroSeconds();
  }

  // end finishing fetch images & labels
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() 
             << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "finishing fetching images...";

  if(this->layer_param_.is_disp_info()) {
    LOG(INFO) << "\nEnd fetching images & labels...\n";
  }

  // visualization of input data
  #ifdef __MULTI_SOURCES_IMAGES_DATA_LAYER_VISUAL__
    const MultiSourcesImagesDataParameter msidp = 
        this->layer_param_.multi_sources_images_data_param();

    const Dtype Zero           = Dtype(0);
    const bool has_visual_path = msidp.has_visual_path();

    if(this->output_labels_ && has_visual_path) {
      const std::string visual_path = msidp.visual_path();
      LOG(INFO) << "visual_path: " << visual_path;
      CreateDir(visual_path.c_str(), 0);
      
      // use default values
      const std::string visual_img_path = visual_path + 
                                          msidp.visual_images_path();
      LOG(INFO) << "visual_images_path: " << visual_img_path;
      CreateDir(visual_img_path.c_str(), 0);

      const Dtype* aux_info2       = this->aux_info_.cpu_data();
      const Dtype* prefetch_label2 = this->prefetch_label_.cpu_data();

      for(int item_id = 0; item_id < this->batch_size_; item_id++) {
        // aux info
        const int ai_offset   = this->aux_info_.offset(item_id);
        const Dtype img_ind   = aux_info2[ai_offset + 0];
        const Dtype width     = aux_info2[ai_offset + 1];
        const Dtype height    = aux_info2[ai_offset + 2];
        const Dtype im_scale  = aux_info2[ai_offset + 3];
        const Dtype flippable = aux_info2[ai_offset + 4];
        const int v_width     = int(width * im_scale);
        const int v_height    = int(height * im_scale);

        std::string caption = flippable ? "missing idxs (flip): " 
                                        : "missing idxs: " ;
                                        
        // labels                                      
        const int label_offset = this->prefetch_label_.offset(item_id);

        // images
        for(int j2 = 0; j2 < this->n_sources_; j2++) {
          cv::Mat img;
          int sc = this->channels_inds_[j2];

          if(this->has_mean_values_) {
            if(this->is_colors_[j2]) {
              img = BlobToColorImage(&(this->prefetch_data_), item_id,
                                     sc, this->mean_values_);
            } else{
              img = BlobToGrayImage(&(this->prefetch_data_),  item_id, 
                                    sc, this->mean_values_);
            }
          // no mean values
          } else {
            if(this->is_colors_[j2]) {
              img = BlobToColorImage(&(this->prefetch_data_), item_id, sc);
            } else {
              img = BlobToGrayImage(&(this->prefetch_data_),  item_id, sc);
            }
          }

          if(j2 > 0) {
            const std::string img_path3 = visual_img_path
                                        + this->imgidxs_[item_id] + "_" 
                                        + this->objidxs_[item_id] + "_ms_"
                                        + to_string(j2)
                                        + this->im_exts_[j2];
            const std::string dire3 = DireName(img_path3);
            CreateDir(dire3.c_str(), 0);
            LOG(INFO) << "img_path3: " << img_path3;                                        
            cv::imwrite(img_path3, img);                                       
            continue;
          }

          // draw labels
          for(int ln = 0; ln < this->label_num_; ln += 2) {
            const int x = int(prefetch_label2[label_offset + ln + 0]);
            const int y = int(prefetch_label2[label_offset + ln + 1]);
            if(x <= Zero || y <= Zero) {
              caption += (to_string(ln) + " ") ;
              continue;
            }
            cv::circle(img, cv::Point(x, y), 
                       2, cv::Scalar(21, 56, 255), 2);
            cv::putText(img, to_string(ln / 2), cv::Point(x - 5 , y),
                        CV_FONT_HERSHEY_COMPLEX,
                        .5, cv::Scalar(255, 56, 21));

            if(this->receptive_field_size_) {
              cv::Mat img2;
              if(this->has_mean_values_) {
                if(this->is_colors_[j2]) {
                  img2 = BlobToColorImage(&(this->prefetch_data_), item_id,
                                         sc, this->mean_values_);
                } else{
                  img2 = BlobToGrayImage(&(this->prefetch_data_),  item_id, 
                                        sc, this->mean_values_);
                }
              // no mean values
              } else {
                if(this->is_colors_[j2]) {
                  img2 = BlobToColorImage(&(this->prefetch_data_), 
                                          item_id, sc);
                } else {
                  img2 = BlobToGrayImage(&(this->prefetch_data_), 
                                         item_id, sc);
                }
              }

              cv::circle(img2, cv::Point(x, y), 2, 
                         cv::Scalar(21, 56, 255), 2);
              cv::putText(img2, to_string(ln / 2), 
                          cv::Point(x - 5 , y), 
                          CV_FONT_HERSHEY_COMPLEX,
                          .5, cv::Scalar(255, 56, 21));

              const int half_rfs = round(this->receptive_field_size_ / 2.0);
              cv::Point p1(max(0, x - half_rfs), 
                           max(0, y - half_rfs));
              cv::Point p2(max(0, x - half_rfs), 
                           min(v_height, y + half_rfs));
              cv::Point p3(min(v_width, x + half_rfs), 
                           min(v_height, y + half_rfs));
              cv::Point p4(min(v_width, x + half_rfs), 
                           max(0, y - half_rfs));

              // img, p1, p2, color, thickness
              cv::line(img2, p1, p2, cv::Scalar(196, 96, 121), 2);
              cv::line(img2, p2, p3, cv::Scalar(196, 96, 121), 2);
              cv::line(img2, p3, p4, cv::Scalar(196, 96, 121), 2);
              cv::line(img2, p4, p1, cv::Scalar(196, 96, 121), 2);

              const std::string img_path2 = visual_img_path 
                                          + this->imgidxs_[item_id] + "_" 
                                          + this->objidxs_[item_id] + "_rf_" 
                                          + to_string(ln / 2) + 
                                          this->im_exts_[j2];
              const std::string dire2 = DireName(img_path2);
              CreateDir(dire2.c_str(), 0);

              cv::imwrite(img_path2, img2);
            }
          }
          cv::putText(img, caption, cv::Point(20, 20), 
                      CV_FONT_HERSHEY_COMPLEX,
                      .5, cv::Scalar(255, 56, 21));

          const std::string img_path = visual_img_path
                                     + this->imgidxs_[item_id] + "_" 
                                     + this->objidxs_[item_id] + "_ms_"
                                     + to_string(j2)
                                     + this->im_exts_[j2];

          LOG(INFO) << "saved img_path: "   << img_path;
          LOG(INFO) << "rows: " << img.rows << ", cols: " << img.cols;
          const std::string dire = DireName(img_path);
          CreateDir(dire.c_str(), 0);
          cv::imwrite(img_path, img);

          LOG(INFO) << "item_id: "          << item_id
                    << ", aux_info count: " << this->aux_info_.count()
                    << ", ai_offset: "      << ai_offset
                    << ", img_ind: "        << img_ind
                    << ", origin width: "   << width 
                    << ", origin height: "  << height
                    << ", im_scales: "      << im_scale
                    << ", flippable: "      << flippable
                    << ", rescale width: "  << img.cols
                    << ", rescale height: " << img.rows;
        }
      }
    }
  #endif
}

template <typename Dtype>
void MultiSourcesImagesDataLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
  // First, join the thread
  DLOG(INFO) << "\nThread joined\n";
  this->JoinPrefetchThread();

  // Used for Accuracy and Visualization layers
  // place in `Forward_cpu` to keep synchronization with top blobs
  // when use `imgidxs` and `images_paths` in other layers
  GlobalVars::set_objidxs(this->objidxs_);
  GlobalVars::set_imgidxs(this->imgidxs_);
  GlobalVars::set_images_paths(this->images_paths_);

  // Reshape to loaded data.
  top[0]->Reshape(this->prefetch_data_.num(), 
                  this->prefetch_data_.channels(),
                  this->prefetch_data_.height(), 
                  this->prefetch_data_.width());
  // Copy the data
  caffe_copy(this->prefetch_data_.count(), 
             this->prefetch_data_.cpu_data(),
             top[0]->mutable_cpu_data());

  DLOG(INFO) << "\nPrefetch copied\n";
  if (this->output_labels_) {
    caffe_copy(this->prefetch_label_.count(), 
               this->prefetch_label_.cpu_data(),
               top[1]->mutable_cpu_data());

    caffe_copy(this->aux_info_.count(), 
               this->aux_info_.cpu_data(),
               top[2]->mutable_cpu_data());
  }
  
  // Start a new prefetch thread
  DLOG(INFO) << "\nCreatePrefetchThread\n";
  this->CreatePrefetchThread();
}

INSTANTIATE_CLASS(MultiSourcesImagesDataLayer);
REGISTER_LAYER_CLASS(MultiSourcesImagesData);

}  // namespace caffe