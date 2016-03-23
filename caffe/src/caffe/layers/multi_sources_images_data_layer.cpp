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

    this->aux_info_.mutable_cpu_data();
    this->prefetch_label_.mutable_cpu_data();
  }

  DLOG(INFO) << "\nInitializing prefetch\n";
  this->CreatePrefetchThread();
  DLOG(INFO) << "\nPrefetch initialized done.\n";
}

// data, labels, im_scales
template <typename Dtype>
void MultiSourcesImagesDataLayer<Dtype>::DataLayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
  CHECK(this->layer_param_.has_multi_sources_images_data_param());
  const MultiSourcesImagesDataParameter msidp = 
      this->layer_param_.multi_sources_images_data_param();

  CHECK(msidp.source_size()      > 0);
  CHECK(msidp.root_folder_size() > 0);
  CHECK(msidp.im_exts_size()     > 0);

  CHECK(msidp.has_batch_size());
  CHECK(msidp.has_is_color());
  CHECK(msidp.has_label_num());
  CHECK(msidp.has_shuffle());
  CHECK(msidp.has_is_scale_image());
  CHECK(msidp.has_parts_orders_path());

  CHECK_EQ(msidp.source_size(), msidp.root_folder_size());
  CHECK_EQ(msidp.source_size(), msidp.im_exts_size());

  this->is_color_             = msidp.is_color();
  this->is_scale_image_       = msidp.is_scale_image();
  this->label_num_            = msidp.label_num();
  this->receptive_field_size_ = 0;

  if(msidp.has_receptive_field_size()) {
    this->receptive_field_size_ = msidp.receptive_field_size();
    CHECK(this->receptive_field_size_ > 0);
  }

  if(this->is_scale_image_) {
    CHECK(msidp.has_max_size());
    CHECK(msidp.min_size_size() > 0);
    CHECK(msidp.has_always_max_size());

    this->max_size_        = msidp.max_size();
    this->always_max_size_ = msidp.always_max_size();

    CHECK_GT(this->max_size_, 0);

    for(int ms = 0; ms < msidp.min_size_size(); ms++) {
      const int min_size = std::min(msidp.min_size(ms), this->max_size_);
      CHECK_GT(min_size, 0);
      this->min_sizes_.push_back(min_size);
    }
  }
  CHECK_EQ(this->label_num_ % 2, 0);

  // check if we want to use mean_value
  CHECK(this->layer_param_.transform_param().has_mean_file() == false) 
      << "Cannot specify mean_file and mean_value at the same time"
      << "\n*** Of cause, here, we restrict that we can't use mean_file. ***";
  this->has_mean_values_ = false;

  const int mean_value_size = 
      this->layer_param_.transform_param().mean_value_size();
  if (mean_value_size > 0) {
    CHECK((mean_value_size == 1 || mean_value_size == 3));
    // opencv: 0: blue; 1: green; 2: red
    for (int c = 0; c < mean_value_size; ++c) {
      const float mean_value = 
          this->layer_param_.transform_param().mean_value(c);
      this->mean_values_.push_back(mean_value);
    }
    if (this->mean_values_.size() == 1) {
      // Replicate the mean_value for simplicity
      for (int c = 1; c < 3; ++c) {
        mean_values_.push_back(mean_values_[0]);
      }
    }
    LOG(INFO) << "\nmean_values: (" 
              << this->mean_values_[0] <<  ", " 
              << this->mean_values_[1] <<  ", " 
              << this->mean_values_[2] <<  ")\n"; 
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
  CHECK_EQ(this->label_num_, this->origin_parts_orders_.size() * 2);
  // key points num
  this->key_points_num_ = this->origin_parts_orders_.size();

  // Read the file with imgidxs and labels
  for(int s = 0; s < msidp.source_size(); s++) {
    const std::string im_ext      = msidp.im_exts(s)
    const std::string source      = msidp.source(s);
    const std::string root_folder = msidp.root_folder(s);

    this->im_exts_.push_back(im_ext);
    this->sources_.push_back(source);
    this->root_folders_.push_back(root_folder);
    vector<std::pair<std::string, std::pair<std::string, 
                   std::vector<float> > > > tmp_line;
    this->lines_.push_back(tmp_line);
  }
  CHECK_EQ(msidp.source_size(), this->lines_.size());

  for(int s = 0; s < msidp.source_size(); s++) {
    const std::string source = this->sources_[s];
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

      const int n_label_info = label_info.size();
      CHECK_GE(n_label_info, 2);

      if(label_info.size() > 2) {
        CHECK(this->output_labels_ == true);
        CHECK_EQ(n_label_info, this->label_num_ + 2);
        CHECK_EQ(s, 0); // only first source file has labels
      } 

      // imgidx & objidx 
      imgidx = label_info[0];
      objidx = label_info[1];
      // labels
      std::vector<float> labels;  // x1, y1, x2, y2, ...
      if(this->output_labels_ && label_info.size() > 2) {
        for(int ln = 2; ln <= this->label_num_ + 1; ln++) {
          pos = std::atof(label_info[ln].c_str());
          labels.push_back(pos);
        }
      }
      // record (source_idx -> (imgidx, (objidx, labels)))
      this->lines_[s].push_back(std::make_pair(imgidx, 
                                   std::make_pair(objidx, labels)));
    }
  }
  // check
  if(this->lines_.size() > 1) {
    for(int j = 1; j < this->lines_.size(); j++) {
      CHECK_EQ(this->lines_[0].size(), this->lines_[j].size());
      for(int j2 = 0; j2 < this->lines_[0].size(); j2++) {
        CHECK_EQ(this->lines_[0][j2].first, this->lines_[j][j2].first);
        CHECK_EQ(this->lines_[0][j2].second.first, 
                 this->lines_[j][j2].second.first);
      }
    }
  }

  // randomly shuffle data
  this->shuffle_ = msidp.shuffle();
  if (this->shuffle_) {
    LOG(INFO) << "Shuffling data";
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    ShuffleImages();
  }
  const int im_n = this->lines_.size() * this->lines_[0].size();
  LOG(INFO) << "A total of " << im_n << " images.";

  // Check if we would need to randomly skip a few data points
  lines_id_ = 0;
  if (msidp.rand_skip()) {
    unsigned int skip = caffe_rng_rand() % msidp.rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    CHECK_GT(this->lines_[0].size(), skip) << "Not enough points to skip";
    lines_id_ = skip;
  }

  // Read an image, and use it to initialize the top blob.
  const int source_idx      = 0;
  const std::string im_path = this->root_folders_[source_idx]  
                            + this->lines_[source_idx][lines_id_].first 
                            + this->im_exts_[source_idx];
  // image 
  cv::Mat img         = ImageRead(im_path, this->is_color_);
  const int max_size  = 100;
  const int n_sources = msidp.source_size();
  const int width     = img.cols > max_size ? max_size : img.cols;
  const int height    = img.rows > max_size ? max_size : img.rows;
  const int channels  = img.channels();
  this->batch_size_   = msidp.batch_size();
  this->crop_size_    = this->layer_param_.transform_param().crop_size();
  CHECK_EQ(this->crop_size_, 0) << "does not need cropping...";
  
  // reshap -> data/images
  top[0]->Reshape(this->batch_size_, channels * n_sources, 
                  height, width);
  this->prefetch_data_.Reshape(this->batch_size_, channels * n_sources, 
                  height, width);
  this->transformed_data_.Reshape(1, channels, height, width);
  
  LOG(INFO) << "\noutput data size: " 
            << top[0]->num()      << ","
            << top[0]->channels() << ","
            << top[0]->height()   << ","
            << top[0]->width()    << "\n";

  // labels & aux info
  if(top.size() > 1) {
    CHECK_EQ(top.size(), 3) 
        << "Reshape: data/images, labels/coordinates, aux info...";

    // reshap -> labels
    top[1]->Reshape(this->batch_size_, this->label_num_, 1, 1);
    this->prefetch_label_.Reshape(this->batch_size_, 
                                  this->label_num_, 1, 1);

    LOG(INFO) << this->label_num_ << ", " << this->batch_size_;
    LOG(INFO) << "output prefetch_label_ size: " << top[1]->count();
    LOG(INFO) << "output prefetch_label_ shape_string: " 
              << top[1]->shape_string();
    
    // reshap -> (img_ind, width, height, im_scale, flippable)
    top[2]->Reshape(this->batch_size_, 5, 1, 1);
    this->aux_info_.Reshape(this->batch_size_, 5, 1, 1);

    LOG(INFO) << "output aux_info_ size: " << top[2]->count();
    LOG(INFO) << "output aux_info_ shape_string: " 
              << top[2]->shape_string();

    // first initialize for create heat maps from coordinates
    // see `heat_maps_from_coords_layer.cpp`
    Dtype* aux_info = top[2]->mutable_cpu_data();
    for(int bs = 0; bs < this->batch_size_; bs++) {
      const int ai_offset     = top[2]->offset(bs);
      aux_info[ai_offset + 0] = Dtype(bs);
      aux_info[ai_offset + 1] = Dtype(width);
      aux_info[ai_offset + 2] = Dtype(height);
      aux_info[ai_offset + 3] = Dtype(1.);
      aux_info[ai_offset + 4] = Dtype(0);
    }
  }

  if(!this->inds_.empty()) {
    this->inds_.clear();
  }
  for(int j = 0; j < this->lines_[0].size(); j++) {
    this->inds_.push_back(j);
  }
}

template <typename Dtype>
void MultiSourcesImagesDataLayer<Dtype>::PerturbedCoordsBias() {
}

// shuffle corresponding indices
template <typename Dtype>
void MultiSourcesImagesDataLayer<Dtype>::ShuffleImages() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(this->inds_.begin(), this->inds_.end(), prefetch_rng);
}

// This function is used to create a thread that prefetches the data.
template <typename Dtype>
void MultiSourcesImagesDataLayer<Dtype>::InternalThreadEntry() {
  // timer
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  CPUTimer timer;

  // initialization && check
  CHECK(this->prefetch_data_.count());
  CHECK(this->transformed_data_.count());
  if(this->output_labels_) {
    CHECK(this->aux_info_.count());
    CHECK(this->prefetch_label_.count());
  }

  // Dtype* prefetch_data = this->prefetch_data_.mutable_cpu_data();
  Dtype* prefetch_label = NULL;
  Dtype* aux_info       = NULL;
  if(this->output_labels_) {
    aux_info = this->aux_info_.mutable_cpu_data();
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

  // get size and scale
  timer.Start();
  const int lines_size = this->lines_[0].size();
  for (int item_id = 0; item_id < this->batch_size_; ++item_id) {
    CHECK_GT(lines_size, lines_id_);

    const int ind             = this->inds_[this->lines_id_];
    CHECK_GT(lines_size, ind);
    const std::string imgidx  = this->lines_[0][ind].first;
    const std::string objidx  = this->lines_[0][ind].second.first;
    const std::string im_path = this->root_folders_[0] + imgidx 
                              + this->im_exts_[0];
    this->objidxs_.push_back(objidx);
    this->imgidxs_.push_back(imgidx);
    this->images_paths_.push_back(im_path);

    cv::Mat im = ImageRead(im_path);
    CHECK(im.data) << "Could not load " << im_path;
    const int width  = im.cols;
    const int height = im.rows;
    widths.push_back(width);
    heights.push_back(height);

    // labels
    if(this->output_labels_) {
      const std::vector<float>& label = 
          this->lines_[0][ind].second.second;
      CHECK_EQ(label.size(), this->label_num_);
      labels.push_back(label);
    }
    
    // go to the next iter
    this->lines_id_++;
    if (this->lines_id_ >= lines_size) {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      this->lines_id_ = 0;
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

  // image scale
  timer.Start();
  if(this->is_scale_image_) {
    for (int item_id = 0; item_id < this->batch_size_; ++item_id) {
      float img_min_size = std::min(widths[item_id], heights[item_id]);
      float img_max_size = std::max(widths[item_id], heights[item_id]);

      float im_scale;
      if(this->always_max_size_) {
        im_scale = float(this->max_size_) / float(img_max_size + 0.0);
      } else {
        const int rand_min_size_idx = 
            this->data_transformer_->Rand(this->min_sizes_.size());
        const int min_size = this->min_sizes_[rand_min_size_idx];
        im_scale = float(std::max(min_size, 
                            int(img_min_size))) / float(std::min(min_size, 
                                int(img_min_size)) + 0.);
        if(im_scale * img_max_size > this->max_size_) {
          im_scale = float(this->max_size_) / float(img_max_size + 0.0);
        }
      }
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
    max_width  = std::max(max_width, 
        int(widths[item_id] * im_scales[item_id]));
    max_height = std::max(max_height, 
        int(heights[item_id] * im_scales[item_id]));
  }

  // reshape & reset
  const int n_sources = this->lines_.size();
  this->prefetch_data_.Reshape(this->batch_size_, 3 * n_sources, 
                               max_height, max_width);
  caffe_set(this->prefetch_data_.count(), Dtype(0), 
            this->prefetch_data_.mutable_cpu_data());
  this->transformed_data_.Reshape(1, 3 * n_sources, max_height, max_width);
  caffe_set(this->transformed_data_.count(), Dtype(0), 
            this->transformed_data_.mutable_cpu_data());
  read_time += timer.MicroSeconds();

  // images (rescale)
  std::vector<bool> do_mirrors;
  for (int item_id = 0; item_id < this->batch_size_; ++item_id) {
    const bool do_mirror = this->layer_param_.transform_param().mirror() 
                        && this->data_transformer_->Rand(2);
    do_mirrors.push_back(do_mirror);

    for(for j2 = 0; j2 < n_sources; j2++) {
      timer.Start();
      const int width           = widths[item_id]  * im_scales[item_id];
      const int height          = heights[item_id] * im_scales[item_id];
      const std::string im_path = this->root_folders_[j2] 
                                + this->this->imgidxs_[item_id] 
                                + this->im_exts_[j2];
      // ####################################################### 
      // need to check <width, height> for all sources???
      // here do not check!!!
      // ToDo            
      // #######################################################                                
      cv::Mat im = ReadImageToCVMat(im_path, height, width, this->is_color_);
      CHECK(im.data) << "Could not load " << im_path;

      if(do_mirror) {
        // >0: horizontal; <0: horizontal&vertical; =0: vertical
        const int flipCode = 1;
        cv::flip(im, im, flipCode);
      }
      
      // 把图片拷贝到放在blob的右上角
      int sc = j2 * 3;
      if(this->has_mean_values_) {
        ImageDataToBlob(&(this->prefetch_data_), item_id, sc, im, 
                          this->mean_values_);
      } else {
        ImageDataToBlob(&(this->prefetch_data_), item_id, sc, im);
      }

      // timer
      read_time += timer.MicroSeconds();
    }
  }
  CHECK_EQ(this->batch_size_, do_mirrors.size());

  // labels (rescale) -- only first source has labels
  for(int item_id = 0; item_id < this->batch_size_; ++item_id) {
    // labels && aux info
    if(this->output_labels_) {
      timer.Start();

      const bool do_mirror = do_mirrors[item_id];
      // labels
      const std::vector<float>& label = labels[item_id];
      CHECK_EQ(label.size(), this->label_num_);
      CHECK_EQ(label.size(), this->key_points_num_ * 2);

      const int label_offset = this->prefetch_label_.offset(item_id);
      for(int kpn = 0; kpn < this->key_points_num_; kpn++) {
        const int id = do_mirror ? this->flippable_parts_orders_[kpn] 
                                 : this->origin_parts_orders_[kpn];
        const int idx = id * 2;
        float x = label[idx];
        x = (do_mirror ? widths[item_id] - x - 1 : x) * im_scales[item_id];
        // y -- height
        float y = label[idx + 1];
        y = y * im_scales[item_id];
        // set
        const int ln = kpn * 2;
        prefetch_label[label_offset + ln + 0] = Dtype(x);
        prefetch_label[label_offset + ln + 1] = Dtype(y);
      }

      // for(int ln = 0; ln < label.size(); ln += 2) {
      //   // here we don't check the whether x&y is valid or not...
      //   // x -- width
      //   float x = label[ln];
      //   x = (do_mirror ? widths[item_id] - x - 1 : x) * im_scales[item_id];
      //   // y -- height
      //   float y = label[ln + 1];
      //   y = y * im_scales[item_id];
      //   // set
      //   prefetch_label[label_offset + ln + 0] = Dtype(x);
      //   prefetch_label[label_offset + ln + 1] = Dtype(y);
      // }

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
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "finishing fetching images...";

  #ifdef __MULTI_SOURCES_IMAGES_DATA_LAYER_VISUAL__
    const MultiSourcesImagesDataParameter msidp = 
        this->layer_param_.multi_sources_images_data_param();

    const Dtype Zero = Dtype(0);
    const bool has_visual_path = msidp.has_visual_path();
    
    if(this->output_labels_ && has_visual_path) {
      const std::string visual_path = msidp.visual_path();
      CreateDir(visual_path.c_str(), 0);
      LOG(INFO) << "visual_path: " << visual_path;

      // use default values
      const std::string visual_img_path = 
          visual_path + msidp.visual_images_path();
      LOG(INFO) << "visual_images_path: " << visual_img_path;
      CreateDir(visual_img_path.c_str(), 0);

      const Dtype* aux_info2       = this->aux_info_.cpu_data();
      const Dtype* prefetch_label2 = this->prefetch_label_.cpu_data();

      for(int item_id = 0; item_id < this->batch_size_; item_id++) {
        const int ai_offset       = this->aux_info_.offset(item_id);
        const Dtype img_ind       = aux_info2[ai_offset + 0];
        const Dtype width         = aux_info2[ai_offset + 1];
        const Dtype height        = aux_info2[ai_offset + 2];
        const Dtype im_scale      = aux_info2[ai_offset + 3];
        const Dtype flippable     = aux_info2[ai_offset + 4];
        const int v_width         = int(width * im_scale);
        const int v_height        = int(height * im_scale);

        for(int j2 = 0; j2 < n_sources; j2++) {
          cv::Mat img;
          const int sc = j2 * 3;
          if(this->has_mean_values_) {
            img = BlobToColorImage(&(this->prefetch_data_), item_id, sc,
                                   this->mean_values_);
          } else {
            img = BlobToColorImage(&(this->prefetch_data_), item_id, sc);
          }
          std::string caption = flippable ? "missing idxs (flip): " 
                                          : "missing idxs: " ;
          if(j2 > 0) {
            const std::string s_im_path = visual_img_path + 
                      + this->imgidxs_[item_id] + "_" 
                      + this->objidxs_[item_id] + "_" 
                      + to_string(j2) 
                      + this->im_exts_[j2];
            continue;
          }

          const int label_offset = this->prefetch_label_.offset(item_id);
          for(int ln = 0; ln < this->label_num_; ln += 2) {
            const int x = int(prefetch_label2[label_offset + ln]);
            const int y = int(prefetch_label2[label_offset + ln + 1]);
            if(x <= Zero || y <= Zero) {
              caption  += (to_string(ln) + " ") ;
              continue;
            }
            cv::circle(img, cv::Point(x, y), 2, 
                       cv::Scalar(21, 56, 255), 2);
            cv::putText(img, to_string(ln / 2), cv::Point(x - 5 , y),
                        CV_FONT_HERSHEY_COMPLEX, .5, 
                        cv::Scalar(255, 56, 21));

            if(this->receptive_field_size_) {
              cv::Mat img2;
              if(this->has_mean_values_) {
                img2 = BlobToColorImage(&(this->prefetch_data_), 
                                        item_id, sc, 
                                        this->mean_values_);
              } else {
                img2 = BlobToColorImage(&(this->prefetch_data_), 
                                        item_id, sc);
              }
              cv::circle(img2, cv::Point(x, y), 2, 
                         cv::Scalar(21, 56, 255), 2);
              cv::putText(img2, to_string(ln / 2), 
                          cv::Point(x - 5 , y), 
                          CV_FONT_HERSHEY_COMPLEX,
                          .5, cv::Scalar(255, 56, 21));

              const int half_rfs = round(this->receptive_field_size_ / 2.0);
              cv::Point p1(max(0,        x - half_rfs), 
                           max(0,        y - half_rfs));
              cv::Point p2(max(0,        x - half_rfs), 
                           min(v_height, y + half_rfs));
              cv::Point p3(min(v_width,  x + half_rfs), 
                           min(v_height, y + half_rfs));
              cv::Point p4(min(v_width,  x + half_rfs), 
                           max(0,        y - half_rfs));
              // img, p1, p2, color, thickness
              cv::line(img2, p1, p2, cv::Scalar(196, 96, 121), 2);
              cv::line(img2, p2, p3, cv::Scalar(196, 96, 121), 2);
              cv::line(img2, p3, p4, cv::Scalar(196, 96, 121), 2);
              cv::line(img2, p4, p1, cv::Scalar(196, 96, 121), 2);

              const std::string img_path2 = visual_img_path + 
                        this->imgidxs_[item_id] + "_"   + 
                        this->objidxs_[item_id] + "_0_" +
                        to_string(ln / 2) + this->im_exts_[j2];
              cv::imwrite(img_path2, img2);
            }
          }
          cv::putText(img, caption, cv::Point(20, 20), 
                      CV_FONT_HERSHEY_COMPLEX,
                      .5, cv::Scalar(255, 56, 21));

          const std::string img_path = visual_img_path + 
                      + this->imgidxs_[item_id] + "_" 
                      + this->objidxs_[item_id] + "_"
                      + to_string(j2) 
                      + this->im_exts_[j2];
          LOG(INFO) << "saved img_path: " << img_path;
          LOG(INFO) << "rows: " << img.rows << ", cols: " << img.cols;
          cv::imwrite(img_path, img);                                          
        }

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
  #endif
}

template <typename Dtype>
void MultiSourcesImagesDataLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
  // First, join the thread
  this->JoinPrefetchThread();
  DLOG(INFO) << "Thread joined";

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
  DLOG(INFO) << "Prefetch copied";
  if (this->output_labels_) {
    caffe_copy(this->prefetch_label_.count(), 
               this->prefetch_label_.cpu_data(),
               top[1]->mutable_cpu_data());
    caffe_copy(this->aux_info_.count(), 
               this->aux_info_.cpu_data(),
               top[2]->mutable_cpu_data());
  }
  // Start a new prefetch thread
  DLOG(INFO) << "CreatePrefetchThread";
  this->CreatePrefetchThread();
}

INSTANTIATE_CLASS(MultiSourcesImagesDataLayer);
REGISTER_LAYER_CLASS(MultiSourcesImagesData);

}  // namespace caffe