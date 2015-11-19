#include <algorithm>
#include <cstring>
#include <vector>

#include "google/protobuf/text_format.h"
#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/pose_estimation_layers.hpp"
#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class SpatialDropoutLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
  SpatialDropoutLayerTest()
      : blob_bottom_(new Blob<Dtype>(5, 3, 4, 6)),
        blob_top_(new Blob<Dtype>()) {
    Caffe::set_random_seed(1701);
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~SpatialDropoutLayerTest() { delete blob_bottom_; delete blob_top_; }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;

  void TestDropoutForward_TRAIN(const float dropout_ratio) {
    LayerParameter layer_param;
    // Fill in the given dropout_ratio, unless it's 0.5, in which case we don't
    // set it explicitly to test that 0.5 is the default.
    SpatialDropoutParameter* spatial_dropout_param = layer_param.mutable_spatial_dropout_param();
    spatial_dropout_param->set_dropout_ratio(dropout_ratio);
    layer_param.set_phase(TRAIN);
    SpatialDropoutLayer<Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    // Now, check values
    const Dtype* bottom_data = this->blob_bottom_->cpu_data();
    const Dtype* top_data = this->blob_top_->cpu_data();
    float scale = 1. / (1. - layer_param.spatial_dropout_param().dropout_ratio());
    CHECK_EQ(this->blob_bottom_->count(), this->blob_top_->count());
    const int count = this->blob_bottom_->count();

    // Initialize num_kept to count the number of inputs NOT dropped out.
    int num_kept = 0;
    for (int i = 0; i < count; ++i) {
      if (top_data[i] != 0) {
        ++num_kept;
        EXPECT_EQ(top_data[i], bottom_data[i] * scale);
      }
    }
    const int mask_count = this->blob_bottom_->num() * this->blob_bottom_->channels();
    const Dtype std_error = sqrt(dropout_ratio * (1 - dropout_ratio) / mask_count);
    // Fail if the number dropped was more than 1.96 * std_error away from the
    // expected number -- requires 95% confidence that the dropout layer is not
    // obeying the given dropout_ratio for test failure.
    const Dtype empirical_dropout_ratio = 1 - num_kept / Dtype(count);
    EXPECT_NEAR(empirical_dropout_ratio, dropout_ratio, 1.96 * std_error);
  }

  void TestDropoutForward_TEST(const float dropout_ratio) {
    // Fill in the given dropout_ratio, unless it's 0.5, in which case we don't
    // set it explicitly to test that 0.5 is the default.
    LayerParameter layer_param;
    SpatialDropoutParameter* spatial_dropout_param = layer_param.mutable_spatial_dropout_param();
    spatial_dropout_param->set_dropout_ratio(dropout_ratio);
    layer_param.set_phase(TEST);
    SpatialDropoutLayer<Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

    // Now, check values
    const Dtype* bottom_data = this->blob_bottom_->cpu_data();
    const Dtype* top_data = this->blob_top_->cpu_data();
    CHECK_EQ(this->blob_bottom_->count(), this->blob_top_->count());
    const int count = this->blob_bottom_->count();
    for (int i = 0; i < count; ++i) {
      EXPECT_EQ(top_data[i], bottom_data[i]);
    }
  }
};

TYPED_TEST_CASE(SpatialDropoutLayerTest, TestDtypesAndDevices);


TYPED_TEST(SpatialDropoutLayerTest, TestDropoutHalf) {
  const float kDropoutRatio = 0.5;
  this->TestDropoutForward_TRAIN(kDropoutRatio);
}

TYPED_TEST(SpatialDropoutLayerTest, TestDropoutThreeQuarters) {
  const float kDropoutRatio = 0.75;
  this->TestDropoutForward_TRAIN(kDropoutRatio);
}

TYPED_TEST(SpatialDropoutLayerTest, TestDropoutHalfTestPhase) {
  const float kDropoutRatio = 0.5;
  this->TestDropoutForward_TEST(kDropoutRatio);
}

TYPED_TEST(SpatialDropoutLayerTest, TestDropoutThreeQuartersTestPhase) {
  const float kDropoutRatio = 0.75;
  this->TestDropoutForward_TEST(kDropoutRatio);
}

TYPED_TEST(SpatialDropoutLayerTest, TestDropoutTestPhase) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  SpatialDropoutParameter* spatial_dropout_param = layer_param.mutable_spatial_dropout_param();
  const float dropout_ratio = 0.5;
  spatial_dropout_param->set_dropout_ratio(dropout_ratio);
  layer_param.set_phase(TEST);
  SpatialDropoutLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Now, check values
  const Dtype* bottom_data = this->blob_bottom_->cpu_data();
  const Dtype* top_data = this->blob_top_->cpu_data();
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    if (top_data[i] != 0) {
      EXPECT_EQ(top_data[i], bottom_data[i]);
    }
  }
}

TYPED_TEST(SpatialDropoutLayerTest, TestDropoutGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  SpatialDropoutParameter* spatial_dropout_param = layer_param.mutable_spatial_dropout_param();
  const float dropout_ratio = 0.5;
  spatial_dropout_param->set_dropout_ratio(dropout_ratio);
  layer_param.set_phase(TRAIN);
  SpatialDropoutLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientEltwise(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(SpatialDropoutLayerTest, TestDropoutGradientTest) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  SpatialDropoutParameter* spatial_dropout_param = layer_param.mutable_spatial_dropout_param();
  const float dropout_ratio = 0.5;
  spatial_dropout_param->set_dropout_ratio(dropout_ratio);
  layer_param.set_phase(TEST);
  SpatialDropoutLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientEltwise(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

}  // namespace caffe
