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

using namespace std;

namespace caffe {

template <typename TypeParam>
class FusionRegionsLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  FusionRegionsLayerTest()
      : cls_sr_inds_(new Blob<Dtype>()),
        primary_regions_(new Blob<Dtype>()),
        secondary_regions_(new Blob<Dtype>()),
        labels_(new Blob<Dtype>()),
        fusion_regions_(new Blob<Dtype>()),
        fusion_regions2_(new Blob<Dtype>()),
        cls_regions_(new Blob<Dtype>()) {}
  // #####################################################################
  // 
  virtual void SetUp() {
    Caffe::set_random_seed(1701);
    // ims_per_batch: 2
    // n_classes: 11
    // n_regions: 5
    // batch_size: 28
    cls_sr_inds_->Reshape(2, 11, 1, 1);
    blob_bottom_vec_.push_back(cls_sr_inds_);
    
    primary_regions_->Reshape(2, 5, 1, 1);
    blob_bottom_vec_.push_back(primary_regions_);

    secondary_regions_->Reshape(28, 5, 1, 1);
    blob_bottom_vec_.push_back(secondary_regions_);

    labels_->Reshape(2, 1, 1, 1);
    blob_bottom_vec_.push_back(labels_);

    blob_top_vec_.push_back(fusion_regions_);
    // 
    propagate_down_.push_back(true);
  }



  // #####################################################################
  // 
  virtual ~FusionRegionsLayerTest() {
    delete cls_sr_inds_;
    delete primary_regions_;
    delete secondary_regions_;
    delete labels_;
    delete fusion_regions_;
    delete fusion_regions2_;
    delete cls_regions_;
  }


  // #####################################################################
  // 
  Blob<Dtype>* const cls_sr_inds_;
  Blob<Dtype>* const primary_regions_;
  Blob<Dtype>* const secondary_regions_;
  Blob<Dtype>* const labels_;
  Blob<Dtype>* const fusion_regions_;
  Blob<Dtype>* const fusion_regions2_;
  Blob<Dtype>* const cls_regions_;

  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
  vector<bool> propagate_down_;



  // #####################################################################
  void TestForwardFusionRegions() {
    int val;
    const int batch_size = 28;
    const int ims_per_batch = 2;
    const int n_classes = 11;
    const int n_regions = 5;
    cls_sr_inds_->Reshape(ims_per_batch, n_classes, 1, 1);
    primary_regions_->Reshape(ims_per_batch, n_regions, 1, 1);
    secondary_regions_->Reshape(batch_size, n_regions, 1, 1);
    labels_->Reshape(ims_per_batch, 1, 1, 1);

    // 
    std::string test_file_dir = "/home/black/caffe/fast-rcnn-action/caffe/src/caffe/test/";
    std::string test_file_sub_dir = "test_data/fusion_regions/";
    // 
    std::string cls_sr_inds_filename = "cls_sr_inds.log";
    std::string cls_sr_inds_filepath = 
        test_file_dir + test_file_sub_dir + cls_sr_inds_filename;
    std::ifstream filer(cls_sr_inds_filepath.c_str());
    CHECK(filer);
    for(int idx = 0; idx < cls_sr_inds_->count(); idx++) {
      filer >> val;
      cls_sr_inds_->mutable_cpu_data()[idx] = val;
    }
    filer.close();
    // 
    std::string primary_regions_filename = "primary_regions.log";
    std::string primary_regions_filepath = 
        test_file_dir + test_file_sub_dir + primary_regions_filename;
    std::ifstream filer2(primary_regions_filepath.c_str());
    CHECK(filer2);
    for(int idx = 0; idx < primary_regions_->count(); idx++) {
      filer2 >> val;
      primary_regions_->mutable_cpu_data()[idx] = val;
    }
    filer2.close();
    // 
    std::string secondary_regions_filename = "secondary_regions.log";
    std::string secondary_regions_filepath = 
        test_file_dir + test_file_sub_dir + secondary_regions_filename;
    std::ifstream filer3(secondary_regions_filepath.c_str());
    CHECK(filer3);
    for(int idx = 0; idx < secondary_regions_->count(); idx++) {
      filer3 >> val;
      secondary_regions_->mutable_cpu_data()[idx] = val;
    }
    filer3.close();
    // labels
    std::string labels_filename = "labels.log";
    std::string labels_filepath = 
        test_file_dir + test_file_sub_dir + labels_filename;
    std::ifstream filer_l(labels_filepath.c_str());
    CHECK(filer_l);
    for(int idx = 0; idx < labels_->count(); idx++) {
      filer_l >> val;
      labels_->mutable_cpu_data()[idx] = val;
    }
    filer_l.close();

    // SetUp
    LayerParameter layer_param;
    FusionRegionsLayer<Dtype> layer(layer_param);
    layer.SetUp(blob_bottom_vec_, blob_top_vec_);

    // 
    EXPECT_EQ(fusion_regions_->num(), ims_per_batch);
    EXPECT_EQ(fusion_regions_->channels(), n_regions);
    EXPECT_EQ(fusion_regions_->height(), 1);
    EXPECT_EQ(fusion_regions_->width(), 1);

    // sr_inds & max_op_idxs
    if (blob_top_vec_.size() > 1) {
      EXPECT_EQ(fusion_regions2_->num(), ims_per_batch * n_classes);
      EXPECT_EQ(fusion_regions2_->channels(), n_regions);
      EXPECT_EQ(fusion_regions2_->height(), 1);
      EXPECT_EQ(fusion_regions2_->width(), 1);      
    }
    // sr_inds & max_op_idxs
    if (blob_top_vec_.size() > 2) {
      EXPECT_EQ(cls_regions_->num(), ims_per_batch * n_classes);
      EXPECT_EQ(cls_regions_->channels(), n_regions);
      EXPECT_EQ(cls_regions_->height(), 1);
      EXPECT_EQ(cls_regions_->width(), 1);
    }

    // ##################################################################
    // Forward
    layer.Forward(blob_bottom_vec_, blob_top_vec_);

    // 
    std::string fusion_regions_filename = "fusion_regions.log";
    std::string fusion_regions_filepath = 
        test_file_dir + test_file_sub_dir + fusion_regions_filename;
    // 
    std::ifstream filer4(fusion_regions_filepath.c_str());
    CHECK(filer4);

    for(int idx = 0; idx < fusion_regions_->count(); idx++) {
      filer4 >> val;
      EXPECT_EQ(fusion_regions_->cpu_data()[idx], val);
    }
    filer4.close();

    // 
    if (blob_top_vec_.size() > 1) {
      std::string fusion_regions2_filename = "fusion_regions2.log";
      std::string fusion_regions2_filepath = 
          test_file_dir + test_file_sub_dir + fusion_regions2_filename;
      // 
      std::ifstream filer4_2(fusion_regions2_filepath.c_str());
      CHECK(filer4_2);

      for(int idx = 0; idx < fusion_regions2_->count(); idx++) {
        filer4_2 >> val;
        EXPECT_EQ(fusion_regions2_->cpu_data()[idx], val);
      }
      filer4_2.close();
    }
    
    if (blob_top_vec_.size() > 2) {
      std::string cls_regions_filename = "cls_regions.log";
      std::string cls_regions_filepath = 
          test_file_dir + test_file_sub_dir + cls_regions_filename;

      std::ifstream filer5(cls_regions_filepath.c_str());
      CHECK(filer5);

      for(int idx = 0; idx < cls_regions_->count(); idx++) {
        filer5 >> val;
        EXPECT_EQ(cls_regions_->cpu_data()[idx], val);
      }
      filer5.close();
    }
  }
};

TYPED_TEST_CASE(FusionRegionsLayerTest, TestDtypesAndDevices);

TYPED_TEST(FusionRegionsLayerTest, TestForward) {
  this->TestForwardFusionRegions();
}

TYPED_TEST(FusionRegionsLayerTest, TestForward2) {
  this->blob_top_vec_.push_back(this->fusion_regions2_);
  this->TestForwardFusionRegions();
}

TYPED_TEST(FusionRegionsLayerTest, TestForward3) {
  this->blob_top_vec_.push_back(this->fusion_regions2_);
  this->blob_top_vec_.push_back(this->cls_regions_);
  this->TestForwardFusionRegions();
}

}  // namespace caffe