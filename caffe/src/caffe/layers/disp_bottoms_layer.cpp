#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/disp_tool_layers.hpp"

namespace caffe {

template <typename Dtype>
void DispBottomsLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>* >& bottom,
      const vector<Blob<Dtype>* >& top) 
{
  // do nothing
  disp_tool_type_ = 0;
}

template <typename Dtype>
void DispBottomsLayer<Dtype>::Reshape(const vector<Blob<Dtype>* >& bottom,
      const vector<Blob<Dtype>*>& top) 
{
  // do nothing
}

template <typename Dtype>
void DispBottomsLayer<Dtype>::disp2screen(const vector<Blob<Dtype>* >& bottom,
      const vector<Blob<Dtype>* >& top) 
{
  for(int i = 0; i < bottom.size(); ++i) {
    for(int n = 0; n < bottom[i]->num(); n++) {
      for(int c = 0; c < bottom[i]->channels(); c++) {
        std::cout << "id: " << i << " n: " << n << " c: " << c << std::endl;
        for(int h = 0; h < bottom[i]->height(); h++) {
          for(int w = 0; w < bottom[i]->width(); w++) {
            std::cout << bottom[i]->data_at(n, c, h, w) << " ";
          } // end w
          std::cout << std::endl;
        } // end h
        std::cout << std::endl;
      } // end c
      std::cout << std::endl;
    } // end n

    std::cout << std::endl;
    std::cout << "layer name: " << this->layer_param_.name() << std::endl;
    std::cout << "id: " << i << std::endl;
    std::cout << " n: " << bottom[i]->num()      << std::endl;
    std::cout << " c: " << bottom[i]->channels() << std::endl;
    std::cout << " h: " << bottom[i]->height()   << std::endl;
    std::cout << " w: " << bottom[i]->width()    << std::endl;
    std::cout << "**********************************************" << std::endl;
    std::cout << std::endl;
    std::cout << std::endl;
  } // end i
  LOG(INFO);
  LOG(INFO);
}

template <typename Dtype>
void DispBottomsLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>* >& bottom,
    const vector<Blob<Dtype>* >& top) 
{
  if(disp_tool_type_ == 0) {
    disp2screen(bottom, top);
  } else {
    NOT_IMPLEMENTED;
  }
}

template <typename Dtype>
void DispBottomsLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>* >& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>* >& bottom) 
{
  const Dtype Zero = Dtype(0);
  CHECK_EQ(propagate_down.size(), bottom.size());

  for(int i = 0; i < propagate_down.size(); ++i) {
    if (propagate_down[i]) { 
      // NOT_IMPLEMENTED; 
      caffe_set(bottom[i]->count(), Zero, bottom[i]->mutable_cpu_diff());
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(DispBottomsLayer);
#endif

INSTANTIATE_CLASS(DispBottomsLayer);
REGISTER_LAYER_CLASS(DispBottoms);

}  // namespace caffe
