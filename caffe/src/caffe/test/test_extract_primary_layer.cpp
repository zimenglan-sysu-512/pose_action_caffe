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
class ExtractPrimaryLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  ExtractPrimaryLayerTest()
      : blob_bottom_(new Blob<Dtype>(12, 3, 6, 5)),
        blob_top_a_(new Blob<Dtype>()),
        blob_top_b_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);

    blob_top_vec_.push_back(blob_top_a_);
    blob_top_vec_.push_back(blob_top_b_);

    propagate_down_.push_back(true);
    propagate_down_.push_back(true);
  }
  virtual ~ExtractPrimaryLayerTest() {
    delete blob_bottom_;
    delete blob_top_a_;
    delete blob_top_b_;
  }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_a_;
  Blob<Dtype>* const blob_top_b_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
  vector<bool> propagate_down_;

  // 
  void TestSetup1() {
    typedef typename TypeParam::Dtype Dtype;
    LayerParameter layer_param;
    const int ims_per_batch = 2;
    layer_param.mutable_extract_primary_param()->set_ims_per_batch(ims_per_batch);
    ExtractPrimaryLayer<Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

    EXPECT_EQ(this->blob_top_a_->num(), 2);
    EXPECT_EQ(this->blob_top_a_->channels(), 3);
    EXPECT_EQ(this->blob_top_a_->height(), 6);
    EXPECT_EQ(this->blob_top_a_->width(), 5);
    // 
    EXPECT_EQ(this->blob_top_b_->num(), 10);
    EXPECT_EQ(this->blob_top_b_->channels(), 3);
    EXPECT_EQ(this->blob_top_b_->height(), 6);
    EXPECT_EQ(this->blob_top_b_->width(), 5);
  }

  // 
  void TestSetup2() {
    typedef typename TypeParam::Dtype Dtype;
    LayerParameter layer_param;
    const int ims_per_batch = 3;
    layer_param.mutable_extract_primary_param()->set_ims_per_batch(ims_per_batch);
    ExtractPrimaryLayer<Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

    EXPECT_EQ(this->blob_top_a_->num(), 3);
    EXPECT_EQ(this->blob_top_a_->channels(), 3);
    EXPECT_EQ(this->blob_top_a_->height(), 6);
    EXPECT_EQ(this->blob_top_a_->width(), 5);
    // 
    EXPECT_EQ(this->blob_top_b_->num(), 9);
    EXPECT_EQ(this->blob_top_b_->channels(), 3);
    EXPECT_EQ(this->blob_top_b_->height(), 6);
    EXPECT_EQ(this->blob_top_b_->width(), 5);
  }
};



// ################################################################



TYPED_TEST_CASE(ExtractPrimaryLayerTest, TestDtypesAndDevices);



// layer_param.mutable_extract_primary_param()->set_axis(0);
TYPED_TEST(ExtractPrimaryLayerTest, TestSetup) {
  this->TestSetup1();
  this->TestSetup2();
}


// where ims_per_batch is 2
TYPED_TEST(ExtractPrimaryLayerTest, Test_2) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  const int ims_per_batch = 2;
  layer_param.mutable_extract_primary_param()->set_ims_per_batch(ims_per_batch);
  ExtractPrimaryLayer<Dtype> layer(layer_param);

  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  int offset = 0;
  int primary_offset = 0;
  int secondary_offset = 0;
  const int feat_count = 
      this->blob_bottom_->count() / this->blob_bottom_->num();
  const int primary_feat_count = 
      this->blob_top_a_->count() / this->blob_top_a_->num();
  const int secondary_feat_count = 
      this->blob_top_b_->count() / this->blob_top_a_->num();
  // 
  EXPECT_EQ(primary_feat_count, feat_count);
  // EXPECT_EQ(secondary_feat_count, feat_count);

  for (int ipb = 0; ipb < ims_per_batch; ipb++) {
    for(int idx = 0; idx < primary_feat_count; idx++) {
      // const Dtype bottom_value = this->blob_bottom_->cpu_data()[offset++];
      // const Dtype top_a_value = this->blob_top_a_->cpu_data()[primary_offset++];
      // EXPECT_EQ(bottom_value, top_a_value);
      EXPECT_EQ(this->blob_bottom_->cpu_data()[offset++],
          this->blob_top_a_->cpu_data()[primary_offset++]);
    }

    for(int idx = 0; idx < secondary_feat_count; idx++) {
      // const Dtype bottom_value = this->blob_bottom_->cpu_data()[offset++];
      // const Dtype top_b_value = this->blob_top_b_->cpu_data()[secondary_offset++];
      // EXPECT_EQ(bottom_value, top_b_value);
      EXPECT_EQ(this->blob_bottom_->cpu_data()[offset++],
          this->blob_top_b_->cpu_data()[secondary_offset++]);
    }
  }

  EXPECT_EQ(primary_offset, this->blob_top_a_->count());
  EXPECT_EQ(secondary_offset, this->blob_top_b_->count());
  EXPECT_EQ(offset, this->blob_bottom_->count());
  EXPECT_EQ(primary_offset + secondary_offset, this->blob_bottom_->count());
}


// where ims_per_batch is 3
TYPED_TEST(ExtractPrimaryLayerTest, Test_3) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  const int ims_per_batch = 3;
  layer_param.mutable_extract_primary_param()->set_ims_per_batch(ims_per_batch);
  ExtractPrimaryLayer<Dtype> layer(layer_param);

  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  int offset = 0;
  int primary_offset = 0;
  int secondary_offset = 0;
  const int feat_count = 
      this->blob_bottom_->count() / this->blob_bottom_->num();
  const int primary_feat_count = 
      this->blob_top_a_->count() / this->blob_top_a_->num();
  const int secondary_feat_count = 
      this->blob_top_b_->count() / this->blob_top_a_->num();
  
  EXPECT_EQ(primary_feat_count, feat_count);
  // EXPECT_EQ(secondary_feat_count, feat_count);


  for (int ipb = 0; ipb < ims_per_batch; ipb++) {
    for(int idx = 0; idx < primary_feat_count; idx++) {
      const Dtype bottom_value = this->blob_bottom_->cpu_data()[offset++];
      const Dtype top_a_value = this->blob_top_a_->cpu_data()[primary_offset++];
      EXPECT_EQ(bottom_value, top_a_value);
    }

    for(int idx = 0; idx < secondary_feat_count; idx++) {
      const Dtype bottom_value = this->blob_bottom_->cpu_data()[offset++];
      const Dtype top_b_value = this->blob_top_b_->cpu_data()[secondary_offset++];
      EXPECT_EQ(bottom_value, top_b_value);
    }
  }

  EXPECT_EQ(primary_offset, this->blob_top_a_->count());
  EXPECT_EQ(secondary_offset, this->blob_top_b_->count());
  EXPECT_EQ(offset, this->blob_bottom_->count());
  EXPECT_EQ(primary_offset + secondary_offset, this->blob_bottom_->count());
  
}


// #############################################################



// where ims_per_batch is 2
TYPED_TEST(ExtractPrimaryLayerTest, TestGradient_2) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  const int ims_per_batch = 2;
  layer_param.mutable_extract_primary_param()->set_ims_per_batch(ims_per_batch);
  ExtractPrimaryLayer<Dtype> layer(layer_param);

  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // 
  FillerParameter filler_param_a;
  GaussianFiller<Dtype> filler_a(filler_param_a);
  filler_a.FillDiff(this->blob_top_a_);

  FillerParameter filler_param_b;
  GaussianFiller<Dtype> filler_b(filler_param_b);
  filler_b.FillDiff(this->blob_top_a_);
  
  layer.Backward(this->blob_top_vec_, this->propagate_down_, this->blob_bottom_vec_);

  int offset = 0;
  int primary_offset = 0;
  int secondary_offset = 0;
  const int feat_count = 
      this->blob_bottom_->count() / this->blob_bottom_->num();
  const int primary_feat_count = 
      this->blob_top_a_->count() / this->blob_top_a_->num();
  const int secondary_feat_count = 
      this->blob_top_b_->count() / this->blob_top_a_->num();

  EXPECT_EQ(primary_feat_count, feat_count);
  // EXPECT_EQ(secondary_feat_count, feat_count);
 
  for (int ipb = 0; ipb < ims_per_batch; ++ipb) {
    for(int idx = 0; idx < primary_feat_count; idx++) {
      const Dtype bottom_value = this->blob_bottom_->cpu_diff()[offset++];
      const Dtype top_a_value = this->blob_top_a_->cpu_diff()[primary_offset++];
      EXPECT_EQ(bottom_value, top_a_value);
    }

    for(int idx = 0; idx < secondary_feat_count; idx++) {
      const Dtype bottom_value = this->blob_bottom_->cpu_diff()[offset++];
      const Dtype top_b_value = this->blob_top_b_->cpu_diff()[secondary_offset++];
      EXPECT_EQ(bottom_value, top_b_value);
    }
  }

  EXPECT_EQ(primary_offset, this->blob_top_a_->count());
  EXPECT_EQ(secondary_offset, this->blob_top_b_->count());
  EXPECT_EQ(offset, this->blob_bottom_->count());
  EXPECT_EQ(primary_offset + secondary_offset, this->blob_bottom_->count());
}


// where ims_per_batch is 3
TYPED_TEST(ExtractPrimaryLayerTest, TestGradient_3) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  const int ims_per_batch = 3;
  layer_param.mutable_extract_primary_param()->set_ims_per_batch(ims_per_batch);
  ExtractPrimaryLayer<Dtype> layer(layer_param);

  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  // 
  FillerParameter filler_param_a;
  GaussianFiller<Dtype> filler_a(filler_param_a);
  filler_a.FillDiff(this->blob_top_a_);

  FillerParameter filler_param_b;
  GaussianFiller<Dtype> filler_b(filler_param_b);
  filler_b.FillDiff(this->blob_top_a_);

  layer.Backward(this->blob_top_vec_, this->propagate_down_, this->blob_bottom_vec_);

  int offset = 0;
  int primary_offset = 0;
  int secondary_offset = 0;
  const int feat_count = 
      this->blob_bottom_->count() / this->blob_bottom_->num();
  const int primary_feat_count = 
      this->blob_top_a_->count() / this->blob_top_a_->num();
  const int secondary_feat_count = 
      this->blob_top_b_->count() / this->blob_top_a_->num();

  EXPECT_EQ(primary_feat_count, feat_count);
  // EXPECT_EQ(secondary_feat_count, feat_count);

  for (int ipb = 0; ipb < ims_per_batch; ++ipb) {
    for(int idx = 0; idx < primary_feat_count; idx++) {
      const Dtype bottom_value = this->blob_bottom_->cpu_diff()[offset++];
      const Dtype top_a_value = this->blob_top_a_->cpu_diff()[primary_offset++];
      EXPECT_EQ(bottom_value, top_a_value);
    }

    for(int idx = 0; idx < secondary_feat_count; idx++) {
      const Dtype bottom_value = this->blob_bottom_->cpu_diff()[offset++];
      const Dtype top_b_value = this->blob_top_b_->cpu_diff()[secondary_offset++];
      EXPECT_EQ(bottom_value, top_b_value);
    }
  }

  EXPECT_EQ(primary_offset, this->blob_top_a_->count());
  EXPECT_EQ(secondary_offset, this->blob_top_b_->count());
  EXPECT_EQ(offset, this->blob_bottom_->count());
  EXPECT_EQ(primary_offset + secondary_offset, this->blob_bottom_->count());
}


// ###################################################################33


// TYPED_TEST(ExtractPrimaryLayerTest, TestGradient2) {
//   typedef typename TypeParam::Dtype Dtype;
//   LayerParameter layer_param;
//   const int ims_per_batch = 2;
//   layer_param.mutable_extract_primary_param()->set_ims_per_batch(ims_per_batch);
//   ExtractPrimaryLayer<Dtype> layer(layer_param);
//   GradientChecker<Dtype> checker(1e-2, 1e-2);
//   checker.CheckGradientEltwise(&layer, this->blob_bottom_vec_,
//       this->blob_top_vec_);
// }


// TYPED_TEST(ExtractPrimaryLayerTest, TestGradient3) {
//   typedef typename TypeParam::Dtype Dtype;
//   LayerParameter layer_param;
//   const int ims_per_batch = 3;
//   layer_param.mutable_extract_primary_param()->set_ims_per_batch(ims_per_batch);
//   ExtractPrimaryLayer<Dtype> layer(layer_param);
//   GradientChecker<Dtype> checker(1e-2, 1e-2);
//   checker.CheckGradientEltwise(&layer, this->blob_bottom_vec_,
//       this->blob_top_vec_);
// }



}  // namespace caffe
