#ifndef CAFFE_PERSON_TORSO_LAYERS_HPP_
#define CAFFE_PERSON_TORSO_LAYERS_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/neuron_layers.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 *        x, if x > 0
 * f(x) = 
 *        alpha * (exp(x) - 1), if x <= 0
 */
template <typename Dtype>
class ELULayer : public NeuronLayer<Dtype> {
 public:
  explicit ELULayer(const LayerParameter& param)
      : NeuronLayer<Dtype>(param) {}

  virtual inline const char* type() const { return "ELU"; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
 private:
  Dtype alpha_;
  Dtype beta_;
};

}  // namespace caffe

#endif  // CAFFE_PERSON_TORSO_LAYERS_HPP_
