#include <opencv2/core/core.hpp>

#include <fstream>   // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "boost/algorithm/string.hpp"
#include "google/protobuf/text_format.h"

#include "caffe/layers/zhouyan_ann_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/common.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/data_layers.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/pose_tool.hpp"
#include "caffe/global_variables.hpp"
#include "caffe/util/math_functions.hpp"

#define __LOAD_DATA_FROM_FILE_LAYER_VISUAL__
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

namespace caffe {

template <typename Dtype>
LoadDataFromFileLayer<Dtype>::~LoadDataFromFileLayer<Dtype>() {
  this->JoinPrefetchThread();
}

template <typename Dtype>
void LoadDataFromFileLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
  BaseDataLayer<Dtype>::LayerSetUp(bottom, top);

  LOG(INFO) << "data/images...";
  this->prefetch_data_.mutable_cpu_data();

  this->output_labels_ = false;
  if(top.size() > 1) {
    CHECK_EQ(top.size(), 2) << "labels...";
    LOG(INFO)               << "labels...";
    this->output_labels_ = true;
    this->prefetch_label_.mutable_cpu_data();
  }

  DLOG(INFO) << "Initializing prefetch";
  this->CreatePrefetchThread();
  DLOG(INFO) << "Prefetch initialized.";
}

template <typename Dtype>
void LoadDataFromFileLayer<Dtype>::DataLayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
  CHECK(this->layer_param_.has_load_data_from_file_param());
  const LoadDataFromFileParameter ldff = 
      this->layer_param_.load_data_from_file_param();

  CHECK(ldff.has_source()); 
  CHECK(ldff.has_root_folder()); 

  this->source_ = ldff.source();
  this->root_folder_ = ldff.root_folder();

  LOG(INFO) << "Opening label file " << this->source_;
  std::ifstream filer(this->source_.c_str());
  CHECK(filer);

  // std::string f1, f2, f3, f4;
  std::string info;
  while (getline(filer, info)) {
    std::vector<std::string> f_info;
    
    boost::trim(info); // f1 f2 f3 [f4]
    boost::split(f_info, info, boost::is_any_of(" "));

    CHECK_GE(f_info.size(), 3);
    CHECK_LE(f_info.size(), 4);

    if(f_info.size() == 3) {
      CHECK(this->output_labels_ == false);
    } else if(f_info.size() == 4) {
      CHECK(this->output_labels_ == true);
    }

    this->lines_.push_back(f_info);
  }
  filer.close();

  this->shuffle_ = ldff.shuffle();
  if (this->shuffle_) {
    LOG(INFO) << "Shuffling data";
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    this->ShuffleImages();
  }
  LOG(INFO) << "\n\nA total of " << this->lines_.size() 
            << " images.\n\n";

  this->lines_id_ = 0;
  if (ldff.rand_skip()) {
    unsigned int skip = caffe_rng_rand() % ldff.rand_skip();
    LOG(INFO) << "Skipping first "       << skip << " data points.";
    CHECK_GT(this->lines_.size(), skip)  << "Not enough points to skip";
    this->lines_id_ = skip;
  }

  this->batch_size_ = ldff.batch_size();
  this->oh_ = ldff.oh();
  this->ow_ = ldff.ow();
  this->on_ = this->oh_ * this->ow_;

  top[0]->Reshape(this->batch_size_, 3, this->oh_, this->ow_);
  this->prefetch_data_.Reshape(this->batch_size_, 3, this->oh_, this->ow_);
  LOG(INFO) << "top 0 blob string (data):  " << top[0]->shape_string();

  if(top.size() > 1 && this->output_labels_) {
    CHECK_EQ(top.size(), 2) 
        << "Reshape: data/images, labels/coordinates, aux info...";
    top[1]->Reshape(this->batch_size_, 1, this->oh_, this->ow_);
    this->prefetch_label_.Reshape(this->batch_size_, 1, this->oh_, this->ow_); 
    LOG(INFO) << "top 1 blob string (label): " << top[1]->shape_string();
  }
  LOG(INFO);

  this->visual_path_ = "";
  if(ldff.has_visual_path()) {
    this->visual_path_ = ldff.visual_path();
    CreateDir(this->visual_path_.c_str(), 0);
  }
  LOG(INFO) << "\nvisual_path: " << this->visual_path_ << "\n";

  this->scale_ = Dtype(1.0);
  if(ldff.has_scale()) {
    this->scale_ = Dtype(ldff.scale());
  }
}

template <typename Dtype>
void LoadDataFromFileLayer<Dtype>::ShuffleImages() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(this->lines_.begin(), this->lines_.end(), prefetch_rng);
}

template <typename Dtype>
void LoadDataFromFileLayer<Dtype>::ReadDataFromFile(
    Blob<Dtype>* blob, const int n, const int c, const std::string path) 
{
  const int height = blob->height();
  const int width  = blob->width();
  CHECK_EQ(height, this->oh_);
  CHECK_EQ(width , this->ow_);

  int h = 0;
  DLOG(INFO) << "Opening label file " << path;
  std::ifstream filer(path.c_str());
  CHECK(filer);

  float v;
  int offset;
  std::string info;
  Dtype* cpu_data = blob->mutable_cpu_data();

  while (getline(filer, info)) {
    h++;
    boost::trim(info);
    std::vector<std::string> data;
    boost::split(data, info, boost::is_any_of(" "));
    CHECK_EQ(data.size(), this->ow_) << path;

    for(int w = 0; w < this->ow_; w++) {
      v = std::atof(data[w].c_str());
      offset = blob->offset(n, c, h, w);
      cpu_data[offset] = Dtype(v);
    }
  }
  filer.close();
  CHECK_EQ(h, this->oh_) << path;
}

template <typename Dtype>
void LoadDataFromFileLayer<Dtype>::VisualDataFromBlob(Blob<Dtype>* blob, 
    const int n, const int c, const std::string path) 
{
  const int height = blob->height();
  const int width  = blob->width();
  CHECK_EQ(height, this->oh_);
  CHECK_EQ(width , this->ow_);

  LOG(INFO) << "Write data into '" << path << "'";

  std::ofstream filer(path.c_str());
  for(int h = 0; h < height; h++) {
    for(int w = 0; w < width - 1; w++) {
      filer << blob->data_at(n, c, h, w) << " ";
    }
    filer << blob->data_at(n, c, h, width - 1) << std::endl;
  }
  filer.close();
}

template <typename Dtype>
void LoadDataFromFileLayer<Dtype>::InternalThreadEntry() {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  CPUTimer timer;

  CHECK(this->prefetch_data_.count());
  if(this->output_labels_) {
    CHECK(this->prefetch_label_.count());
  }

  this->prefetch_data_.Reshape(this->batch_size_, 
                               3, this->oh_, this->ow_);
  caffe_set(this->prefetch_data_.count(), 
            Dtype(0), 
            this->prefetch_data_.mutable_cpu_data());
  
  if(this->output_labels_) {
    this->prefetch_label_.Reshape(this->batch_size_, 
                                  1, this->oh_, this->ow_);
    caffe_set(this->prefetch_label_.count(), 
              Dtype(0), 
              this->prefetch_label_.mutable_cpu_data());
  }
  read_time += timer.MicroSeconds();

  if(this->layer_param_.is_disp_info()) {
    LOG(INFO) << "Start fetching images & labels...";
  }

  // fetch data and labels
  std::vector<std::vector<std::string> > paths;
  for (int item_id = 0; item_id < this->batch_size_; ++item_id) {
    timer.Start();
    if(this->layer_param_.is_disp_info()) {
      LOG(INFO) << "Start fetching images & labels -- " 
                << item_id << "...";
    }

    const std::vector<std::string >& ps = lines_[this->lines_id_];
    CHECK_GE(ps.size(), 3);
    CHECK_LE(ps.size(), 4);
    paths.push_back(ps);

    for(int j = 0; j < ps.size(); j++) {
      const std::string p = ps[j];
      if(this->layer_param_.is_disp_info()) {
        LOG(INFO) << "j: " << j <<  " -- data file: " << p;
      }

      if(j >= 3) {
        CHECK_EQ(this->output_labels_, true);
        this->ReadDataFromFile(&(this->prefetch_label_), item_id, 0, p);
      } else {
        this->ReadDataFromFile(&(this->prefetch_data_),  item_id, j, p);
      }
    }

    this->lines_id_++;
    if (this->lines_id_ >= this->lines_.size()) {
      DLOG(INFO) << "Restarting data prefetching from start.";
      this->lines_id_ = 0;
      if (this->shuffle_) {
        this->ShuffleImages();
      }
    }

    // timer
    read_time += timer.MicroSeconds();
  }
  CHECK_EQ(paths.size(), this->batch_size_);

  // end finishing fetch images & labels
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() 
             << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "finishing fetching images...";

  if(this->layer_param_.is_disp_info()) {
    LOG(INFO) << "\nEnd fetching images & labels...\n";
  }

  if(this->visual_path_.length() > 0) {
    for (int item_id = 0; item_id < this->batch_size_; ++item_id) {
      const std::vector<std::string >& ps = paths[item_id];
      CHECK_GE(ps.size(), 3);
      CHECK_LE(ps.size(), 4);

      for(int j = 0; j < ps.size(); j++) {
        const std::string p  = ps[j];
        const std::string p2 = this->visual_path_ + FileName(p);
        if(this->layer_param_.is_disp_info()) {
          LOG(INFO) << "j: " << j <<  " -- visual file: " << p2;
        }

        if(j >= 3) {
          CHECK_EQ(this->output_labels_, true);
          this->VisualDataFromBlob(&(this->prefetch_label_), item_id, 0, p2);
        } else {
          this->VisualDataFromBlob(&(this->prefetch_data_),  item_id, j, p2);
        }
      }
    }
  }
}

template <typename Dtype>
void LoadDataFromFileLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
  // First, join the thread
  DLOG(INFO) << "\nThread joined\n";
  this->JoinPrefetchThread();

  // Reshape to loaded data.
  top[0]->Reshape(this->prefetch_data_.num(), 
                  this->prefetch_data_.channels(),
                  this->prefetch_data_.height(), 
                  this->prefetch_data_.width());
  caffe_scal(this->prefetch_data_.count(), this->scale_, 
             this->prefetch_data_.mutable_cpu_data());
  // Copy the data
  caffe_copy(this->prefetch_data_.count(), 
             this->prefetch_data_.cpu_data(),
             top[0]->mutable_cpu_data());
  if(this->layer_param_.is_disp_info()) {
    LOG(INFO) << "Copy and Reshape top 0 blob (data)...";
  }

  DLOG(INFO) << "\nPrefetch copied\n";
  if (this->output_labels_) {
    top[1]->Reshape(this->prefetch_label_.num(), 
                    this->prefetch_label_.channels(),
                    this->prefetch_label_.height(), 
                    this->prefetch_label_.width());
    caffe_scal(this->prefetch_label_.count(), this->scale_, 
               this->prefetch_label_.mutable_cpu_data());
    caffe_copy(this->prefetch_label_.count(), 
               this->prefetch_label_.cpu_data(),
               top[1]->mutable_cpu_data());
    if(this->layer_param_.is_disp_info()) {
      LOG(INFO) << "Copy and Reshape top 1 blob (labels)...";
    }
  }
  
  // Start a new prefetch thread
  DLOG(INFO) << "\nCreatePrefetchThread\n";
  this->CreatePrefetchThread();
}

INSTANTIATE_CLASS(LoadDataFromFileLayer);
REGISTER_LAYER_CLASS(LoadDataFromFile);

}  // namespace caffe