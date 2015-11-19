#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

using namespace std;

namespace caffe {

template <typename Dtype>
void SigmoidCrossEntropyLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  sigmoid_bottom_vec_.clear();
  sigmoid_bottom_vec_.push_back(bottom[0]);
  sigmoid_top_vec_.clear();
  sigmoid_top_vec_.push_back(sigmoid_output_.get());
  sigmoid_layer_->SetUp(sigmoid_bottom_vec_, sigmoid_top_vec_);

  ///
  label_blob_.Reshape(
      bottom[0]->num(), 
      bottom[0]->channels(), 
      bottom[0]->height(), 
      bottom[0]->width()
  );
}

template <typename Dtype>
void SigmoidCrossEntropyLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);

  ///
  label_blob_.Reshape(
      bottom[0]->num(), 
      bottom[0]->channels(), 
      bottom[0]->height(), 
      bottom[0]->width()
  );
  const int sub_count1 = bottom[1]->count() / bottom[1]->num();
  const int sub_count2 = bottom[0]->count() / bottom[1]->num();
  if(sub_count1 < sub_count2) {
    const Dtype Ones = Dtype(1);
    const Dtype Zeros = Dtype(0);
    caffe_set(
        label_blob_.count(), 
        Zeros, 
        label_blob_.mutable_cpu_data()
    );
    // copy the label
    for(int n = 0; n < label_blob_.num(); n++) {
      const int offset1 = bottom[1]->offset(n);
      const int offset2 = label_blob_.offset(n);
      const int sub_count = bottom[1]->count() / bottom[1]->num();
      // 
      for(int idx = 0; idx < sub_count; idx++) {
        const int label_idx = bottom[1]->cpu_data()[offset1 + idx];
        label_blob_.mutable_cpu_data()[offset2 + label_idx] = Ones;
      }
    }
  } else {
    CHECK_EQ(bottom[0]->count(), bottom[1]->count()) <<
        "SIGMOID_CROSS_ENTROPY_LOSS layer inputs must have the same count.";
    caffe_copy(label_blob_.count(), bottom[1]->cpu_data(), label_blob_.mutable_cpu_data());
  }
  // int ind = 0, ind2 = 0;
  // for(int n = 0; n < label_blob_.num(); n++) {
  //   for(int idx = 0; idx < bottom[1]->count() / bottom[1]->num(); idx++) {
  //     std::cout << " " << bottom[1]->cpu_data()[ind++];
  //   }
  //   std::cout << std::endl;
  //   for(int idx = 0; idx < label_blob_.count() / label_blob_.num(); idx++) {
  //     std::cout << " " << label_blob_.cpu_data()[ind2++];
  //   }
  //   std::cout << std::endl;
  // }
  // std::cout << std::endl;

  ///
  sigmoid_layer_->Reshape(sigmoid_bottom_vec_, sigmoid_top_vec_);
}

template <typename Dtype>
void SigmoidCrossEntropyLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the sigmoid outputs.
  sigmoid_bottom_vec_[0] = bottom[0];
  sigmoid_layer_->Forward(sigmoid_bottom_vec_, sigmoid_top_vec_);
  // Compute the loss (negative log likelihood)
  const int count = bottom[0]->count();
  const int num = bottom[0]->num();
  // Stable version of loss computation from input data
  const Dtype* input_data = bottom[0]->cpu_data();

  // const Dtype* target = bottom[1]->cpu_data();
  const Dtype* target = label_blob_.cpu_data();

  Dtype loss = 0;
  for (int i = 0; i < count; ++i) {
    loss -= input_data[i] * (target[i] - (input_data[i] >= 0)) -
        log(1 + exp(input_data[i] - 2 * input_data[i] * (input_data[i] >= 0)));
  }
  top[0]->mutable_cpu_data()[0] = loss / num;
}

template <typename Dtype>
void SigmoidCrossEntropyLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    // First, compute the diff
    const int count = bottom[0]->count();
    const int num = bottom[0]->num();
    const Dtype* sigmoid_output_data = sigmoid_output_->cpu_data();

    // const Dtype* target = bottom[1]->cpu_data();
    const Dtype* target = label_blob_.cpu_data();

    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    caffe_sub(count, sigmoid_output_data, target, bottom_diff);
    // Scale down gradient
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    caffe_scal(count, loss_weight / num, bottom_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU(SigmoidCrossEntropyLossLayer);
#endif

INSTANTIATE_CLASS(SigmoidCrossEntropyLossLayer);
REGISTER_LAYER_CLASS(SigmoidCrossEntropyLoss);

}  // namespace caffe
