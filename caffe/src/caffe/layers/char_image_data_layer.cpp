#include <opencv2/core/core.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "boost/algorithm/string.hpp"
#include "google/protobuf/text_format.h"

#include "caffe/char_recognition_layers.hpp"
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

#define __CHAR_DATA_LAYER_VISUAL__
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

namespace caffe {

template <typename Dtype>
CharImageDataLayer<Dtype>::~CharImageDataLayer<Dtype>() {
  this->JoinPrefetchThread();
}

template <typename Dtype>
void CharImageDataLayer<Dtype>::LayerSetUp(
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
    CHECK_LE(top.size(), 3) << "labels/coordinates, aux info..."; 
    LOG(INFO) << "labels/coordinates, aux info...";
    this->prefetch_label_.mutable_cpu_data();
    this->aux_info_.mutable_cpu_data();
  }

  DLOG(INFO) << "Initializing prefetch";
  this->CreatePrefetchThread();
  DLOG(INFO) << "Prefetch initialized.";
}

// data, labels, im_scales
template <typename Dtype>
void CharImageDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) 
{
  CHECK(this->layer_param_.has_char_image_data_param());
  const CharImageDataParameter char_image_data_param = 
      this->layer_param_.char_image_data_param();
  CHECK(char_image_data_param.source_size() > 0);
  CHECK(char_image_data_param.root_folder_size() > 0);
  CHECK_EQ(char_image_data_param.source_size(), char_image_data_param.root_folder_size());

  CHECK(char_image_data_param.has_batch_size());
  CHECK(char_image_data_param.has_is_color());
  CHECK(char_image_data_param.has_label_num());
  CHECK(char_image_data_param.has_img_ext());
  CHECK(char_image_data_param.has_shuffle());
  CHECK(char_image_data_param.has_is_scale_image());

  this->is_color_  = char_image_data_param.is_color();
  this->is_scale_image_ = char_image_data_param.is_scale_image();
  this->label_num_ = char_image_data_param.label_num();
  this->img_ext_ = char_image_data_param.img_ext();
  if(this->is_scale_image_) {
    CHECK(char_image_data_param.has_max_size());
    CHECK(char_image_data_param.min_size_size() > 0);
    CHECK(char_image_data_param.has_always_max_size());

    this->max_size_ = char_image_data_param.max_size();
    this->always_max_size_ = char_image_data_param.always_max_size();
    CHECK_GT(this->max_size_, 0);
    for(int ms = 0; ms < char_image_data_param.min_size_size(); ms++) {
      const int min_size = std::min(char_image_data_param.min_size(ms), this->max_size_);
      CHECK_GT(min_size, 0);
      this->min_sizes_.push_back(min_size);
    }
  }
  CHECK_EQ(this->label_num_, 1);

  // check if we want to use mean_value
  CHECK(this->layer_param_.transform_param().has_mean_file() == false) 
      << "Cannot specify mean_file and mean_value at the same time"
      << "\n*** Of cause, here, we restrict that we can't use mean_file. ***";
  this->has_mean_values_ = false;
  const int mean_value_size = this->layer_param_.transform_param().mean_value_size();
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
    LOG(INFO) << "mean_values: (" 
        << this->mean_values_[0] <<  ", " 
        << this->mean_values_[1] <<  ", " 
        << this->mean_values_[2] <<  ")"; 
    this->has_mean_values_ = true;
  }

  // Read the file with imgidxs and labels
  for(int s = 0; s < char_image_data_param.source_size(); s++) {
    const std::string source = char_image_data_param.source(s);
    const std::string root_folder = char_image_data_param.root_folder(s);
    this->sources_.push_back(source);
    this->root_folders_.push_back(root_folder);
  }

  for(int s = 0; s < char_image_data_param.source_size(); s++) {
    const std::string source = this->sources_[s];
    const std::string root_folder = this->root_folders_[s];

    LOG(INFO) << "Opening file " << source;
    std::ifstream infile(source.c_str());
    CHECK(infile);

    int label;
    string objidx;
    string imgidx;
    string line_info;
    while (getline(infile, line_info)) {
      std::vector<std::string> label_info;
      // imgidx objidx label_id
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
      label = std::atoi(label_info[2].c_str());
      // record (source_idx, (imgidx, (objidx, label)))
      lines_.push_back(std::make_pair(s, std::make_pair(imgidx, std::make_pair(objidx, label))));
    }
  }
  // randomly shuffle data
  this->shuffle_ = this->layer_param_.char_image_data_param().shuffle();
  if (this->shuffle_) {
    LOG(INFO) << "Shuffling data";
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    ShuffleImages();
  }
  LOG(INFO) << "A total of " << lines_.size() << " images.";

  // Check if we would need to randomly skip a few data points
  lines_id_ = 0;
  if (this->layer_param_.char_image_data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
        this->layer_param_.char_image_data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    CHECK_GT(lines_.size(), skip) << "Not enough points to skip";
    lines_id_ = skip;
  }

  // Read an image, and use it to initialize the top blob.
  const int source_idx = lines_[lines_id_].first;
  const std::string image_path = 
      this->root_folders_[source_idx] + lines_[lines_id_].second.first + this->img_ext_;
  // image 
  cv::Mat img = ImageRead(image_path, this->is_color_);
  const int max_size = 100;
  const int width = img.cols > max_size ? max_size : img.cols;
  const int height = img.rows > max_size ? max_size : img.rows;
  const int channels = img.channels();
  this->crop_size_ = this->layer_param_.transform_param().crop_size();
  this->batch_size_ = this->layer_param_.char_image_data_param().batch_size();
  CHECK_EQ(this->crop_size_, 0) << "does not need cropping...";
  
  // data/images
  top[0]->Reshape(this->batch_size_, channels, height, width);
  this->prefetch_data_.Reshape(this->batch_size_, channels, height, width);
  this->transformed_data_.Reshape(1, channels, height, width);
  
  LOG(INFO) << "output data size: " 
      << top[0]->num() << "," << top[0]->channels() << "," << top[0]->height() << "," << top[0]->width();

  // labels & aux info
  if(top.size() > 1) {
    CHECK_LE(top.size(), 3) << "Reshape: data/images, labels/coordinates, aux info...";
    // label
    top[1]->Reshape(this->batch_size_, this->label_num_, 1, 1);
    this->prefetch_label_.Reshape(this->batch_size_, this->label_num_, 1, 1);
    LOG(INFO) << this->label_num_ << ", " << this->batch_size_;
    LOG(INFO) << "output prefetch_label_ size: " << top[1]->count();
    LOG(INFO) << "output prefetch_label_ shape_string: " << top[1]->shape_string();
    // (img_ind, width, height, im_scale, flippable)
    this->aux_info_.Reshape(this->batch_size_, 5, 1, 1);
    LOG(INFO) << "output aux_info_ shape_string: " << this->aux_info_.shape_string();
    if(top.size() > 2) {
      top[2]->Reshape(this->batch_size_, 5, 1, 1);
      LOG(INFO) << "output aux_info_ shape_string: " << top[2]->shape_string();
      LOG(INFO) << "output aux_info_ size: " << top[2]->count();
      Dtype* aux_info = top[2]->mutable_cpu_data();
      for(int bs = 0; bs < this->batch_size_; bs++) {
        const int aux_info_offset = top[2]->offset(bs);
          aux_info[aux_info_offset + 0] = Dtype(bs);
          aux_info[aux_info_offset + 1] = Dtype(width);
          aux_info[aux_info_offset + 2] = Dtype(height);
          aux_info[aux_info_offset + 3] = Dtype(1.);
          aux_info[aux_info_offset + 4] = Dtype(0);
      }
    }
  }
}

template <typename Dtype>
void CharImageDataLayer<Dtype>::ShuffleImages() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

// This function is used to create a thread that prefetches the data.
template <typename Dtype>
void CharImageDataLayer<Dtype>::InternalThreadEntry() {
  // Initialize
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  CPUTimer timer;
  CHECK(this->prefetch_data_.count());
  CHECK(this->transformed_data_.count());
  if(this->output_labels_) {
    CHECK(this->prefetch_label_.count());
    CHECK(this->aux_info_.count());
  }

  // Dtype* prefetch_data = this->prefetch_data_.mutable_cpu_data();
  Dtype* prefetch_label = NULL;
  Dtype* aux_info = NULL;
  if(this->output_labels_) {
    prefetch_label = this->prefetch_label_.mutable_cpu_data();
    aux_info = this->aux_info_.mutable_cpu_data();
  }

  // variables
  std::vector<int> labels;
  std::vector<int> widths;
  std::vector<int> heights;
  std::vector<float> im_scales;

  // clear before setting values
  if(this->objidxs_.size() > 0) this->objidxs_.clear();
  CHECK_EQ(this->objidxs_.size(), 0);
  if(this->imgidxs_.size() > 0) this->imgidxs_.clear();
  CHECK_EQ(this->imgidxs_.size(), 0);
  if(this->images_paths_.size() > 0) this->images_paths_.clear();
  CHECK_EQ(this->images_paths_.size(), 0);

  // get size and scale
  timer.Start();
  const int lines_size = lines_.size();
  for (int item_id = 0; item_id < this->batch_size_; ++item_id) {
    CHECK_GT(lines_size, lines_id_);

    const int source_idx = lines_[lines_id_].first;
    const std::string imgidx = lines_[lines_id_].second.first;
    const std::string objidx = lines_[lines_id_].second.second.first;
    const std::string image_path = this->root_folders_[source_idx] + imgidx + this->img_ext_;
    this->objidxs_.push_back(objidx);
    this->imgidxs_.push_back(imgidx);
    this->images_paths_.push_back(image_path);

    cv::Mat cv_img = ImageRead(image_path);
    CHECK(cv_img.data) << "Could not load " << image_path;
    const int width = cv_img.cols;
    const int height = cv_img.rows;
    widths.push_back(width);
    heights.push_back(height);

    // labels
    if(this->output_labels_) {
      const int label = lines_[lines_id_].second.second.second;
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
  CHECK_EQ(widths.size(), this->batch_size_);
  CHECK_EQ(heights.size(), this->batch_size_);
  CHECK_EQ(this->objidxs_.size(), this->batch_size_);
  CHECK_EQ(this->imgidxs_.size(), this->batch_size_);
  CHECK_EQ(this->images_paths_.size(), this->batch_size_);
  if(this->output_labels_) {
    CHECK_EQ(labels.size(), this->batch_size_);
  }
  read_time += timer.MicroSeconds();

  // image scale
  timer.Start();
  if(this->is_scale_image_) {
    for (int item_id = 0; item_id < this->batch_size_; ++item_id) {
      float im_scale;
      float img_min_size = std::min(widths[item_id], heights[item_id]);
      float img_max_size = std::max(widths[item_id], heights[item_id]);
      if(this->always_max_size_) {
        im_scale = float(this->max_size_) / float(img_max_size + 0.0);
      } else {
        const int rand_min_size_idx = this->data_transformer_->Rand(this->min_sizes_.size());
        const int min_size = this->min_sizes_[rand_min_size_idx];
        im_scale = float(std::max(min_size, int(img_min_size))) / float(std::min(min_size, int(img_min_size)) + 0.);
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
  // Find the max height and max width along all the images alone
  int max_height = -1;
  int max_width = -1;
  for (int item_id = 0; item_id < this->batch_size_; ++item_id) {
    max_width = std::max(max_width, int(widths[item_id] * im_scales[item_id]));
    max_height = std::max(max_height, int(heights[item_id] * im_scales[item_id]));
  }

  // Reshape & Reset
  this->prefetch_data_.Reshape(this->batch_size_, 3, max_height, max_width);
  caffe_set(this->prefetch_data_.count(), Dtype(0), this->prefetch_data_.mutable_cpu_data());
  this->transformed_data_.Reshape(1, 3, max_height, max_width);
  caffe_set(this->transformed_data_.count(), Dtype(0), this->transformed_data_.mutable_cpu_data());
  read_time += timer.MicroSeconds();

  // images (rescale)
  for (int item_id = 0; item_id < this->batch_size_; ++item_id) {
    timer.Start();
    const int width = widths[item_id] * im_scales[item_id];
    const int height = heights[item_id] * im_scales[item_id];
    const std::string image_path = this->images_paths_[item_id];
    const bool do_mirror = this->layer_param_.transform_param().mirror() && this->data_transformer_->Rand(2);

    cv::Mat cv_img = ReadImageToCVMat(image_path, height, width, this->is_color_);
    CHECK(cv_img.data) << "Could not load " << image_path;
    if(do_mirror) {
      // >0: horizontal; <0: horizontal&vertical; =0: vertical
      const int flipCode = 1;
      cv::flip(cv_img, cv_img, flipCode);
    }
    // 把图片拷贝到放在blob的右上角
    if(this->has_mean_values_) {
      ImageDataToBlob(&(this->prefetch_data_), item_id, cv_img, this->mean_values_);
    } else {
      ImageDataToBlob(&(this->prefetch_data_), item_id, cv_img);
    }
    read_time += timer.MicroSeconds();

    // labels && aux info
    if(this->output_labels_) {
      timer.Start();
      // labels
      const int label = labels[item_id];
      const int label_offset = this->prefetch_label_.offset(item_id);
      prefetch_label[label_offset] = Dtype(label);
      // aux info
      const int aux_info_offset = this->aux_info_.offset(item_id);
      aux_info[aux_info_offset + 0] = Dtype(item_id);
      aux_info[aux_info_offset + 1] = Dtype(widths[item_id]);
      aux_info[aux_info_offset + 2] = Dtype(heights[item_id]);
      aux_info[aux_info_offset + 3] = Dtype(im_scales[item_id]);  
      aux_info[aux_info_offset + 4] = do_mirror ? Dtype(1) : Dtype(0);  
    }
    read_time += timer.MicroSeconds();
  }

  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "finishing fetching images...";

  #ifdef __CHAR_DATA_LAYER_VISUAL__
    const bool has_visual_path = 
            this->layer_param_.char_image_data_param().has_visual_path();
    if(this->output_labels_ && has_visual_path) {
      const std::string visual_path = 
          this->layer_param_.char_image_data_param().visual_path();
      CreateDir(visual_path.c_str(), 0);
      // use default values
      const std::string visual_img_path = visual_path + 
          this->layer_param_.char_image_data_param().visual_images_path();
      CreateDir(visual_img_path.c_str(), 0);

      const Dtype* prefetch_label2 = this->prefetch_label_.cpu_data();
      for(int item_id = 0; item_id < this->batch_size_; item_id++) {
        cv::Mat img;
        if(this->has_mean_values_) {
          img = BlobToColorImage(&(this->prefetch_data_), item_id, this->mean_values_);
        } else {
          img = BlobToColorImage(&(this->prefetch_data_), item_id);
        }
        // const Dtype* aux_info2 = this->aux_info_.cpu_data();
        // const int aux_info_offset = this->aux_info_.offset(item_id);
        // const Dtype img_ind   = aux_info2[aux_info_offset + 0];
        // const Dtype width     = aux_info2[aux_info_offset + 1];
        // const Dtype height    = aux_info2[aux_info_offset + 2];
        // const Dtype im_scale  = aux_info2[aux_info_offset + 3];
        // const Dtype flippable = aux_info2[aux_info_offset + 4];
        const int label_offset = this->prefetch_label_.offset(item_id);
        const int label = int(prefetch_label2[label_offset]);
        std::string caption = to_string(label);
        cv::putText(img, caption, cv::Point(5, 5), CV_FONT_HERSHEY_COMPLEX,
            .5, cv::Scalar(255, 56, 21));
        const std::string img_path = visual_img_path + 
            this->imgidxs_[item_id] + "_" + this->objidxs_[item_id] + this->img_ext_;
        cv::imwrite(img_path, img);
      }
    }
  #endif
}

template <typename Dtype>
void CharImageDataLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
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
  top[0]->Reshape(this->prefetch_data_.num(), this->prefetch_data_.channels(),
      this->prefetch_data_.height(), this->prefetch_data_.width());
  // Copy the data
  caffe_copy(this->prefetch_data_.count(), this->prefetch_data_.cpu_data(),
             top[0]->mutable_cpu_data());
  DLOG(INFO) << "Prefetch copied";
  if (this->output_labels_) {
    caffe_copy(this->prefetch_label_.count(), this->prefetch_label_.cpu_data(),
               top[1]->mutable_cpu_data());
    if(top.size() > 2) {
      caffe_copy(this->aux_info_.count(), this->aux_info_.cpu_data(),
                 top[2]->mutable_cpu_data());
    }
  }
  // Start a new prefetch thread
  DLOG(INFO) << "CreatePrefetchThread";
  this->CreatePrefetchThread();
}

INSTANTIATE_CLASS(CharImageDataLayer);
REGISTER_LAYER_CLASS(CharImageData);

}  // namespace caffe