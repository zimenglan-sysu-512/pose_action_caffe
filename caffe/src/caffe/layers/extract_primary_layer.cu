#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/fast_rcnn_action_layers.hpp"


namespace caffe {


template <typename Dtype>
void ExtractPrimaryLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) 
{
  const int bottom_num = bottom[0]->num();
  const int bottom_count = bottom[0]->count();
  // 
  const int top_count1 = top[0]->count();
  const int top_count2 = top[1]->count();
  CHECK_EQ(bottom_count, top_count1 + top_count2)
      << "the size of input must equal to the size of outputs";

  const int bottom_sub_count = bottom_count / bottom_num;
  const int top_sub_count1 = top[0]->count() / top[0]->num();
  // Note
  const int top_sub_count2 = top[1]->count() / top[0]->num();
  CHECK_EQ(bottom_sub_count, top_sub_count1);
  
  // 
  Dtype* top_data1 = top[0]->mutable_gpu_data();
  Dtype* top_data2 = top[1]->mutable_gpu_data();
  const Dtype* bottom_data = bottom[0]->gpu_data();

  int offset = 0;
  int top_offset1 = 0;
  int top_offset2 = 0;
  for(int ipb = 0; ipb < this->ims_per_batch_; ipb++) {
    // primary region
    caffe_copy(
        top_sub_count1, 
        bottom_data + offset,
        top_data1 + top_offset1
    );
    offset += top_sub_count1;
    top_offset1 += top_sub_count1; 

    // secondary region sets
    caffe_copy(
        top_sub_count2,
        bottom_data + offset,
        top_data2 + top_offset2
    );
    offset += top_sub_count2;
    top_offset2 += top_sub_count2;
  }
  
  CHECK_EQ(offset, bottom_count);
  CHECK_EQ(top_offset1, top_count1);
  CHECK_EQ(top_offset2, top_count2);
}



template <typename Dtype>
void ExtractPrimaryLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) 
{
  if(!propagate_down[0]) return;

  const int bottom_num = bottom[0]->num();
  const int bottom_count = bottom[0]->count();
  // 
  const int top_count1 = top[0]->count();
  const int top_count2 = top[1]->count();
  CHECK_EQ(bottom_count, top_count1 + top_count2)
      << "the size of input must equal to the size of outputs";

  const int bottom_sub_count = bottom_count / bottom_num;
  const int top_sub_count1 = top[0]->count() / top[0]->num();
  // Note
  const int top_sub_count2 = top[1]->count() / top[0]->num();
  CHECK_EQ(bottom_sub_count, top_sub_count1);

  const Dtype* top_diff1 = top[0]->gpu_diff();
  const Dtype* top_diff2 = top[1]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();

  // 
  int offset = 0;
  int top_offset1 = 0;
  int top_offset2 = 0;
  for(int ipb = 0; ipb < this->ims_per_batch_; ipb++) {
    // primary region
    caffe_copy(
        top_sub_count1, 
        top_diff1 + top_offset1,
        bottom_diff + offset
    );
    offset += top_sub_count1;
    top_offset1 += top_sub_count1; 

    // secondary region sets
    caffe_copy(
        top_sub_count2,
        top_diff2 + top_offset2,
        bottom_diff + offset
    );
    offset += top_sub_count2;
    top_offset2 += top_sub_count2;
  }
  
  CHECK_EQ(offset, bottom_count);
  CHECK_EQ(top_offset1, top_count1);
  CHECK_EQ(top_offset2, top_count2);
}


INSTANTIATE_LAYER_GPU_FUNCS(ExtractPrimaryLayer);


}  // namespace caffe
