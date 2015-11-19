#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/fast_rcnn_action_layers.hpp"


namespace caffe {



template <typename Dtype>
void ExtractPrimaryLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) 
{
  if(bottom.size() == 1) {
    CHECK(this->layer_param_.has_extract_primary_param())
        << "must has extract_primary_param";
    // 
    const ExtractPrimaryParameter& extract_primary_param = 
        this->layer_param_.extract_primary_param();

    CHECK(extract_primary_param.has_ims_per_batch())
        << "ims_per_batch should be specified (the number of images per batch)";

    CHECK_GT(extract_primary_param.ims_per_batch(), 0)
        << "ims_per_batch should be specified (the number of images per batch)";
  } else if(bottom.size() == 2) {
    CHECK_EQ(bottom[1]->num(), bottom[0]->num());
    CHECK_EQ(bottom[1]->channels(),  1);
    CHECK_EQ(bottom[1]->height(),  1);
    CHECK_EQ(bottom[1]->width(),  1);
    // 
    const int n_secondary_regions = bottom[1]->cpu_data()[0];
    for(int idx = 0; idx < bottom[1]->count(); idx++) {
      CHECK_EQ(n_secondary_regions, bottom[1]->cpu_data()[idx]);
    }
  } else {
    LOG(FATAL) << "wrong prototxt settings";
  }
}



// top[0]: primary regions
// top[1]: corrsponding secondary region sets
template <typename Dtype>
void ExtractPrimaryLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) 
{
  const int num = bottom[0]->num();

  if(bottom.size() == 1) {
    const ExtractPrimaryParameter& extract_primary_param = 
        this->layer_param_.extract_primary_param();
    this->ims_per_batch_ = extract_primary_param.ims_per_batch();
  } else {
    // record the number of secondary regions of one image
    // num_batch = ims_per_batch * (1 + n_secondary_regions)
    // 为了防止空data的时候初始化网络
    const int n_secondary_regions = int(bottom[1]->cpu_data()[0]);
    this->ims_per_batch_ = num / (n_secondary_regions + 1);
  }

  CHECK_LE(this->ims_per_batch_, num)
      << "ims_per_batch should be less than or equal to `num`";
  CHECK(!(num % this->ims_per_batch_))
      << "`num == k * ims_per_batch` is true, where k = 1, 2, ...";
  
  //
  const int channels = bottom[0]->channels();
  const int height = bottom[0]->height();
  const int width = bottom[0]->width();

  const int top_num1 = this->ims_per_batch_;
  const int top_num2 = num - this->ims_per_batch_;
  // CHECK_GT(top_num2, 0) << "Top has two blobs";

  top[0]->Reshape(top_num1, channels, height, width);
  if(top_num2) {
    top[1]->Reshape(top_num2, channels, height, width);
  } else {
    // 为了防止空data的时候初始化网络
    top[1]->Reshape(1, channels, height, width);
  }
}



template <typename Dtype>
void ExtractPrimaryLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
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
  Dtype* top_data1 = top[0]->mutable_cpu_data();
  Dtype* top_data2 = top[1]->mutable_cpu_data();
  const Dtype* bottom_data = bottom[0]->cpu_data();

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
void ExtractPrimaryLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
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

  const Dtype* top_diff1 = top[0]->cpu_diff();
  const Dtype* top_diff2 = top[1]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

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




#ifdef CPU_ONLY
STUB_GPU(ExtractPrimaryLayer);
#endif


INSTANTIATE_CLASS(ExtractPrimaryLayer);
REGISTER_LAYER_CLASS(ExtractPrimary);


}  // namespace caffe
