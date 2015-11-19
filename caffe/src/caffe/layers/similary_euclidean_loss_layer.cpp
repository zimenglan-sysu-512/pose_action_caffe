  #include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/fast_rcnn_action_layers.hpp"

namespace caffe {

template <typename Dtype>
void SimilaryEuclideanLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
  // here we restrict the input batch size is 2
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->num(), 2) 
      << "only for the primary_fc7 of input, which consists of two input images";
  CHECK_EQ(bottom[1]->num(), 2)
    << "only for the labels of input, which consists of two input images";
  CHECK_EQ(bottom[1]->count(), 2)
    << "only for the labels of input, which consists of two input images";

  const int channels = bottom[0]->channels();
  const int height = bottom[0]->height();
  const int width = bottom[0]->width();
  // Reshape diff_
  diff_.Reshape(1, channels, height, width);
  CHECK_EQ(bottom[0]->count(), diff_.count() * 2);

  CHECK(this->layer_param_.has_similary_euclidean_loss_param())
        << "must has similary_euclidean_loss_param or use default values...";
  const SimilaryEuclideanLossParameter similary_euclidean_loss_param = 
      this->layer_param_.similary_euclidean_loss_param();
  this->learn_lr_ = similary_euclidean_loss_param.learn_lr();
  this->loss_thresh_ = similary_euclidean_loss_param.loss_thresh();
  this->loss_bias_ = similary_euclidean_loss_param.loss_bias();
}


template <typename Dtype>
void SimilaryEuclideanLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) 
{
  const int num = bottom[0]->num();
  const int count = bottom[0]->count();
  const int sub_count = count / num;
  CHECK_EQ(diff_.count(), sub_count);
  // 
  const int label1 = int(bottom[1]->cpu_data()[0]);
  const int label2 = int(bottom[1]->cpu_data()[1]);
  const Dtype flag = (label1 == label2 ? Dtype(1) : Dtype(-1));

  // loss_func = 1/2 * I(L(x), L(y)) * {alpha * (F(x) - F(y))^2 - bias}
  // where L(x) is the label of x, F(x) is the feature of x, and F(x) is normalized
  //  I(,) is the sign funtcion, if L(x) is equal to L(y), return 1, otherwise return -1
  // feature may comes from fc7
  const Dtype* bottom_data = bottom[0]->cpu_data();
  caffe_sub(
      sub_count,
      bottom_data,
      bottom_data + sub_count,
      diff_.mutable_cpu_data()
  );
  // compute the loss
  Dtype dot = caffe_cpu_dot(
      sub_count, 
      diff_.cpu_data(), 
      diff_.cpu_data()
  );
  Dtype loss = this->learn_lr_ * dot - Dtype(this->loss_bias_);
  loss = Dtype(0.5) * flag * loss;

  // if the loss is less or equal to loss threshold, we think they are
  // extremely the same or different, otherwise we need to refresh them
  top[0]->mutable_cpu_data()[0] = loss;
  this->above_loss_thresh_ = (loss >= this->loss_thresh_);

  // 
  const Dtype scal_factor = flag * this->learn_lr_;
  caffe_scal(
      sub_count,
      scal_factor,
      diff_.mutable_cpu_data()
  );
  // LOG(INFO) << "label1: " << label1 << ", label2: " << label2 << ", flag: "
  //     << flag << ", loss: " << loss;
  // LOG(INFO) << "(learn_lr: " << this->learn_lr_ << ", loss_bias: "
  //     << this->loss_bias_ << ", loss_thresh: " << this->loss_thresh_
  //     << ")";
}


template <typename Dtype>
void SimilaryEuclideanLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) 
{
  const int num = bottom[0]->num();
  const int count = bottom[0]->count();
  const int sub_count = count / num;

  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  caffe_set(count, Dtype(0), bottom_diff);

  for (int i = 0; i < bottom[0]->num(); ++i) {
    if (propagate_down[i] && this->above_loss_thresh_) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
      caffe_cpu_axpby(
          sub_count,              // count
          alpha,                  // alpha
          diff_.cpu_data(),       // a
          Dtype(0),               // beta
          bottom_diff             // b
      );  
    }
    // add offset
    bottom_diff += sub_count;
  }
}


#ifdef CPU_ONLY
STUB_GPU(SimilaryEuclideanLossLayer);
#endif


INSTANTIATE_CLASS(SimilaryEuclideanLossLayer);
REGISTER_LAYER_CLASS(SimilaryEuclideanLoss);


}  // namespace caffe
