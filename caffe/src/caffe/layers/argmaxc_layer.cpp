#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/pose_estimation_layers.hpp"

namespace caffe {

template <typename Dtype>
void ArgMaxCLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
  out_max_val_ = this->layer_param_.arg_max_c_param().out_max_val();
}

template <typename Dtype>
void ArgMaxCLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
  const int height = bottom[0]->height();
  const int width  = bottom[0]->width();
  if (out_max_val_) { // Produces max_ind and max_val
    top[0]->Reshape(bottom[0]->num(), 2, height, width);
  } else {            // Produces only max_ind
    top[0]->Reshape(bottom[0]->num(), 1, height, width);
  }
}

template <typename Dtype>
void ArgMaxCLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data          = top[0]->mutable_cpu_data();

  const int num      = bottom[0]->num();
  const int channels = bottom[0]->channels();
  const int height   = bottom[0]->height();
  const int width    = bottom[0]->width();

  int ind  = -1;
  Dtype v1 = Dtype(0);
  Dtype v2 = Dtype(0);

  for (int n = 0; n < num; ++n) {
    for(int h = 0; h < height; ++h) {
      for(int w = 0; w < width; ++w) {
        ind = 0;
        v1  = bottom_data[bottom[0]->offset(n, 0, h, w)];
        for(int c = 1; c < channels; ++c){
          v2  = bottom_data[bottom[0]->offset(n, c, h, w)];
          if(v2 > v1) {
            v1  = v2;
            ind = c;
          }
        } // end c
        top_data[top[0]->offset(n, 0, h, w)] = Dtype(ind);
        if(out_max_val_) {
          top_data[top[0]->offset(n, 1, h, w)] = v1;
        }
      } // end w
    } // end h
  } // end n
}

template <typename Dtype>
void ArgMaxCLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
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

INSTANTIATE_CLASS(ArgMaxCLayer);
REGISTER_LAYER_CLASS(ArgMaxC);

}  // namespace caffe
