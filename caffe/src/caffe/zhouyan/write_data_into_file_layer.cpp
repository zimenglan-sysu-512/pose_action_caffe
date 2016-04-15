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
#include "caffe/util/pose_tool.hpp"
#include "caffe/global_variables.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void WriteDataIntoFileLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
  CHECK(this->layer_param_.has_write_data_into_file_param());

  this->scale_ = Dtype(1);
  // WriteDataIntoFileParameter write_data_into_file_param
  if(this->layer_param_.write_data_into_file_param().has_scale()) {
    this->scale_ = 
        Dtype(this->layer_param_.write_data_into_file_param().scale());
  }
 
  this->visual_dire_ = 
      this->layer_param_.write_data_into_file_param().visual_dire();
  CreateDir(this->visual_dire_.c_str(), 0);

  this->im_c_ = 0;
  this->n_images_ = 
      this->layer_param_.write_data_into_file_param().n_images();
  CHECK_GT(this->n_images_, 0) << this->layer_param_.name();

  this->file_ext_ = ".txt";
  if(this->layer_param_.write_data_into_file_param().has_file_ext()) {
    this->file_ext_ = 
        this->layer_param_.write_data_into_file_param().file_ext();
  }
}

template <typename Dtype>
void WriteDataIntoFileLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
  CHECK_EQ(bottom[0]->channels(), 1);
  top[0]->Reshape(bottom[0]->num(), bottom[0]->channels(),
                  bottom[0]->height(), bottom[0]->width());
}

template <typename Dtype>
void WriteDataIntoFileLayer<Dtype>::WriteResults(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
  const int num      = top[0]->num();
  const int channels = top[0]->channels();
  const int height   = top[0]->height();
  const int width    = top[0]->width();
  CHECK_EQ(channels, 1);

  for(int n = 0; n < num; n++) {
    this->im_c_++;
    
    std::string name = "test_" + to_string(this->im_c_) + 
                        this->file_ext_;
    std::string path = this->visual_dire_ + name;
    LOG(INFO) << "im_c: " << this->im_c_ << " "
              << "path: " << path        << "\n";

    std::ofstream filer(path.c_str());
    for(int h = 0; h < height; h++) {
      for(int w = 0; w < width - 1; w++) {
        filer << top[0]->data_at(n, 0, h, w) << " ";
      }
      filer << top[0]->data_at(n, 0, h, width - 1) << std::endl;
    }
    filer.close();

    this->im_c_ = this->im_c_ % this->n_images_;
  }
}


template <typename Dtype>
void WriteDataIntoFileLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
  CHECK_EQ(bottom[0]->count(), top[0]->count());
  caffe_copy(bottom[0]->count(), bottom[0]->cpu_data(),
             top[0]->mutable_cpu_data());
  caffe_scal(top[0]->count(), this->scale_, top[0]->mutable_cpu_data());

  this->WriteResults(bottom, top);
}

template <typename Dtype>
void WriteDataIntoFileLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) 
{
  const Dtype zero = Dtype(0);
  CHECK_EQ(propagate_down.size(), bottom.size());

  for (int i = 0; i < propagate_down.size(); ++i) {
    if (propagate_down[i]) { 
      // NOT_IMPLEMENTED; 
      // caffe_set(bottom[i]->count(), zero, bottom[i]->mutable_cpu_diff());
      caffe_copy(top[i]->count(), top[i]->cpu_diff(), 
                 bottom[i]->mutable_cpu_diff());
      caffe_scal(bottom[i]->count(), this->scale_, 
                 bottom[i]->mutable_cpu_diff());
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(WriteDataIntoFileLayer);
#endif

INSTANTIATE_CLASS(WriteDataIntoFileLayer);
REGISTER_LAYER_CLASS(WriteDataIntoFile);

}  // namespace caffe