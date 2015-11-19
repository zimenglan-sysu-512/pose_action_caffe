#include <cstring>
#include <string>
#include <vector>

#include "google/protobuf/text_format.h"
#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/insert_splits.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/fast_rcnn_action_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class CrossBlobsSumLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  CrossBlobsSumLayerTest()
      : blob_bottom_0_(new Blob<Dtype>(2, 3, 6, 5)),
        blob_bottom_1_(new Blob<Dtype>(2, 3, 6, 5)),
        blob_bottom_2_(new Blob<Dtype>(2, 3, 6, 5)),
        blob_top_(new Blob<Dtype>())  
  {
    FillerParameter filler_param;
    shared_ptr<ConstantFiller<Dtype> > filler;

    filler_param.set_value(1.);
    filler.reset(new ConstantFiller<Dtype>(filler_param));
    filler->Fill(this->blob_bottom_0_);

    filler_param.set_value(2.);
    filler.reset(new ConstantFiller<Dtype>(filler_param));
    filler->Fill(this->blob_bottom_1_);

    filler_param.set_value(3.);
    filler.reset(new ConstantFiller<Dtype>(filler_param));
    filler->Fill(this->blob_bottom_2_);

    blob_bottom_vec_0_.push_back(blob_bottom_0_);
    blob_bottom_vec_0_.push_back(blob_bottom_1_);
    blob_bottom_vec_1_.push_back(blob_bottom_0_);
    blob_bottom_vec_1_.push_back(blob_bottom_2_);

    blob_top_vec_.push_back(blob_top_);

    propagate_down_.push_back(true);
    propagate_down_.push_back(true);
  }

  virtual ~CrossBlobsSumLayerTest() {
    delete blob_bottom_0_; 
    delete blob_bottom_1_;
    delete blob_bottom_2_; 
    delete blob_top_;
  }

  Blob<Dtype>* const blob_bottom_0_;
  Blob<Dtype>* const blob_bottom_1_;
  Blob<Dtype>* const blob_bottom_2_;
  Blob<Dtype>* const blob_top_;

  vector<Blob<Dtype>*> blob_bottom_vec_0_, blob_bottom_vec_1_;
  vector<Blob<Dtype>*> blob_top_vec_;

  vector<bool> propagate_down_;
};



// ################################################################



TYPED_TEST_CASE(CrossBlobsSumLayerTest, TestDtypesAndDevices);



TYPED_TEST(CrossBlobsSumLayerTest, TestSetup_0) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  CrossBlobsSumLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_0_, this->blob_top_vec_);

  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 3);
  EXPECT_EQ(this->blob_top_->height(), 6);
  EXPECT_EQ(this->blob_top_->width(), 5);
}



TYPED_TEST(CrossBlobsSumLayerTest, TestSetup_1) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  CrossBlobsSumLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_1_, this->blob_top_vec_);

  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 3);
  EXPECT_EQ(this->blob_top_->height(), 6);
  EXPECT_EQ(this->blob_top_->width(), 5);
}



// where ims_per_batch is 2
TYPED_TEST(CrossBlobsSumLayerTest, Test_0) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  CrossBlobsSumLayer<Dtype> layer(layer_param);

  layer.SetUp(this->blob_bottom_vec_0_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_0_, this->blob_top_vec_);

  const int count = this->blob_top_->count();

  for (int idx = 0; idx < count; idx++) {
    // 
    Dtype sum = Dtype(0);

    for(int bv = 0; bv < this->blob_bottom_vec_0_.size(); bv++) {
      sum += this->blob_bottom_vec_0_[bv]->cpu_data()[idx];
    }

    const Dtype top_value = this->blob_top_->cpu_data()[idx];
    EXPECT_EQ(sum, top_value);
  }
}


// where ims_per_batch is 2
TYPED_TEST(CrossBlobsSumLayerTest, Test_1) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  CrossBlobsSumLayer<Dtype> layer(layer_param);

  layer.SetUp(this->blob_bottom_vec_1_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_1_, this->blob_top_vec_);

  const int count = this->blob_top_->count();

  for (int idx = 0; idx < count; idx++) {
    // 
    Dtype sum = Dtype(0);

    for(int bv = 0; bv < this->blob_bottom_vec_1_.size(); bv++) {
      sum += this->blob_bottom_vec_1_[bv]->cpu_data()[idx];
    }

    const Dtype top_value = this->blob_top_->cpu_data()[idx];
    EXPECT_EQ(sum, top_value);
  }
}



// #############################################################


TYPED_TEST(CrossBlobsSumLayerTest, TestGradient_0) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  CrossBlobsSumLayer<Dtype> layer(layer_param);

  layer.SetUp(this->blob_bottom_vec_0_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_0_, this->blob_top_vec_);

  // 
  FillerParameter filler_param_a;
  GaussianFiller<Dtype> filler_a(filler_param_a);
  filler_a.FillDiff(this->blob_top_);

  layer.Backward(this->blob_top_vec_, this->propagate_down_, this->blob_bottom_vec_0_);

  const int count = this->blob_top_->count();

  for (int idx = 0; idx < count; idx++) {
    // 
    const Dtype top_diff_value = this->blob_top_->cpu_diff()[idx];
    for(int bv = 0; bv < this->blob_bottom_vec_0_.size(); bv++) {
      const Dtype bottom_diff_value = this->blob_bottom_vec_0_[bv]->cpu_diff()[idx];
      EXPECT_EQ(bottom_diff_value, top_diff_value);
    }
  }
}



TYPED_TEST(CrossBlobsSumLayerTest, TestGradient_1) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  CrossBlobsSumLayer<Dtype> layer(layer_param);

  layer.SetUp(this->blob_bottom_vec_1_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_1_, this->blob_top_vec_);

  // 
  FillerParameter filler_param_a;
  GaussianFiller<Dtype> filler_a(filler_param_a);
  filler_a.FillDiff(this->blob_top_);

  layer.Backward(this->blob_top_vec_, this->propagate_down_, this->blob_bottom_vec_1_);

  const int count = this->blob_top_->count();

  for (int idx = 0; idx < count; idx++) {
    // 
    const Dtype top_diff_value = this->blob_top_->cpu_diff()[idx];
    for(int bv = 0; bv < this->blob_bottom_vec_1_.size(); bv++) {
      const Dtype bottom_diff_value = this->blob_bottom_vec_1_[bv]->cpu_diff()[idx];
      EXPECT_EQ(bottom_diff_value, top_diff_value);
    }
  }

}



}  // namespace caffe
