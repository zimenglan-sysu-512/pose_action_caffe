#ifndef CAFFE_DISP_TOOL_LAYERS_HPP_
#define CAFFE_DISP_TOOL_LAYERS_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/neuron_layers.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class DispBottomsLayer : public NeuronLayer<Dtype> {
 public:
  explicit DispBottomsLayer(const LayerParameter& param)
      : NeuronLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>* >& bottom,
      const vector<Blob<Dtype>* >& top);
  virtual void Reshape(const vector<Blob<Dtype>* >& bottom,
      const vector<Blob<Dtype>* >& top);
  virtual inline const char* type() const { return "DispBottoms"; }
  virtual inline int MinBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 0; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>* >& bottom,
      const vector<Blob<Dtype>* >& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

 private:
  virtual void disp2screen(const vector<Blob<Dtype>* >& bottom,
      const vector<Blob<Dtype>* >& top);

 private:
  int disp_tool_type_;
  std::string visual_dire_;
  std::string visual_sub_dire_;
};

}  // namespace caffe

#endif  // CAFFE_DISP_TOOL_LAYERS_HPP_
