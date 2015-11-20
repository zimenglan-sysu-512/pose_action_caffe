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
class ExtractPrimaryLayerTest2 : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  ExtractPrimaryLayerTest2()
      : blob_bottom_(new Blob<Dtype>(12, 2, 3, 2)),
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
  virtual ~ExtractPrimaryLayerTest2() {
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
    EXPECT_EQ(this->blob_top_a_->channels(), 2);
    EXPECT_EQ(this->blob_top_a_->height(), 3);
    EXPECT_EQ(this->blob_top_a_->width(), 2);
    // 
    EXPECT_EQ(this->blob_top_b_->num(), 10);
    EXPECT_EQ(this->blob_top_b_->channels(), 2);
    EXPECT_EQ(this->blob_top_b_->height(), 3);
    EXPECT_EQ(this->blob_top_b_->width(), 2);
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
    EXPECT_EQ(this->blob_top_a_->channels(), 2);
    EXPECT_EQ(this->blob_top_a_->height(), 3);
    EXPECT_EQ(this->blob_top_a_->width(), 2);
    // 
    EXPECT_EQ(this->blob_top_b_->num(), 9);
    EXPECT_EQ(this->blob_top_b_->channels(), 2);
    EXPECT_EQ(this->blob_top_b_->height(), 3);
    EXPECT_EQ(this->blob_top_b_->width(), 2);
  }


  // where ims_per_batch is 2
  void TestForwardBackward2()  {
    typedef typename TypeParam::Dtype Dtype;
    LayerParameter layer_param;
    const int ims_per_batch = 2;
    const int num = 12;
    const int channels = 2;
    const int height = 3;
    const int width = 2;
    layer_param.mutable_extract_primary_param()->set_ims_per_batch(ims_per_batch);
    ExtractPrimaryLayer<Dtype> layer(layer_param);


    // ###############################################################
    // init data values
    std::string test_file_dir = 
        "/home/black/caffe/fast-rcnn-action/caffe/src/caffe/test/";
    std::string test_file_sub_dir = "test_data/extract_primary/";
    std::string bottom_data_input_filename = "bottom_data_input.log";
    std::string bottom_data_input_filepath = 
        test_file_dir + test_file_sub_dir + bottom_data_input_filename;
    // 
    std::ifstream filer(bottom_data_input_filepath.c_str());
    CHECK(filer);
    // 
    int val;
    for(int idx = 0; idx < blob_bottom_->count(); idx++) {
      filer >> val;
      blob_bottom_->mutable_cpu_data()[idx] = val;
    }
    filer.close();


    // ###############################################################
    // SetUp
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    // 
    EXPECT_EQ(this->blob_top_a_->num(), ims_per_batch);
    EXPECT_EQ(this->blob_top_a_->channels(), channels);
    EXPECT_EQ(this->blob_top_a_->height(), height);
    EXPECT_EQ(this->blob_top_a_->width(), width);
    // 
    EXPECT_EQ(this->blob_top_b_->num(), num - ims_per_batch);
    EXPECT_EQ(this->blob_top_b_->channels(), channels);
    EXPECT_EQ(this->blob_top_b_->height(), height);
    EXPECT_EQ(this->blob_top_b_->width(), width);


    // ###############################################################
    // Forward
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);


    // ###############################################################
    // Check blob_top_a_
    std::string top_data_output_2_filename_a = "top_data_output_2_a.log";
    std::string top_data_output_2_filepath_a = 
        test_file_dir + test_file_sub_dir + top_data_output_2_filename_a;
    // 
    std::ifstream filer1(top_data_output_2_filepath_a.c_str());
    CHECK(filer1);

    for(int idx = 0; idx < blob_top_a_->count(); idx++) {
      filer1 >> val;
      EXPECT_EQ(blob_top_a_->cpu_data()[idx], val);
    }
    filer1.close();

    // Check blob_top_b_
    std::string top_data_output_2_filename_b = "top_data_output_2_b.log";
    std::string top_data_output_2_filepath_b = 
        test_file_dir + test_file_sub_dir + top_data_output_2_filename_b;
    // 
    std::ifstream filer2(top_data_output_2_filepath_b.c_str());
    CHECK(filer2);

    for(int idx = 0; idx < blob_top_b_->count(); idx++) {
      filer2 >> val;
      EXPECT_EQ(blob_top_b_->cpu_data()[idx], val);
    }
    filer2.close();


    // ###############################################################
    // init diff values
    std::string top_diff_input_2_filename_a = "top_diff_input_2_a.log";
    std::string top_diff_input_2_filepath_a = 
        test_file_dir + test_file_sub_dir + top_diff_input_2_filename_a;

    std::ifstream filer3(top_diff_input_2_filepath_a.c_str());
    CHECK(filer3);

    for(int idx = 0; idx < blob_top_a_->count(); idx++) {
      filer3 >> val;
      blob_top_a_->mutable_cpu_diff()[idx] = val;
    }
    filer3.close();
    //
    std::string top_diff_input_2_filename_b = "top_diff_input_2_b.log";
    std::string top_diff_input_2_filepath_b = 
        test_file_dir + test_file_sub_dir + top_diff_input_2_filename_b;

    std::ifstream filer4(top_diff_input_2_filepath_b.c_str());
    CHECK(filer4);

    for(int idx = 0; idx < blob_top_b_->count(); idx++) {
      filer4 >> val;
      blob_top_b_->mutable_cpu_diff()[idx] = val;
    }
    filer4.close();




    // ###############################################################
    // Backward
    layer.Backward(blob_top_vec_, propagate_down_, blob_bottom_vec_);


    // 
    // Check
    std::string bottom_diff_input_2_filename = "bottom_diff_output_2.log";
    std::string bottom_diff_input_2_filepath = 
        test_file_dir + test_file_sub_dir + bottom_diff_input_2_filename;

    std::ifstream filer5(bottom_diff_input_2_filepath.c_str());
    CHECK(filer5);

    for(int idx = 0; idx < blob_bottom_->count(); idx++) {
      filer5 >> val;
      blob_bottom_->mutable_cpu_diff()[idx] = val;
    }
    filer5.close();
  }

  // where ims_per_batch is 3
  void TestForwardBackward3()  {
    typedef typename TypeParam::Dtype Dtype;
    LayerParameter layer_param;
    const int ims_per_batch = 3;
    const int num = 12;
    const int channels = 2;
    const int height = 3;
    const int width = 2;
    layer_param.mutable_extract_primary_param()->set_ims_per_batch(ims_per_batch);
    ExtractPrimaryLayer<Dtype> layer(layer_param);


    // ###############################################################
    // init data values
    std::string test_file_dir = 
        "/home/black/caffe/fast-rcnn-action/caffe/src/caffe/test/";
    std::string test_file_sub_dir = "test_data/extract_primary/";
    std::string bottom_data_input_filename = "bottom_data_input.log";
    std::string bottom_data_input_filepath = 
        test_file_dir + test_file_sub_dir + bottom_data_input_filename;
    // 
    std::ifstream filer(bottom_data_input_filepath.c_str());
    CHECK(filer);
    // 
    int val;
    for(int idx = 0; idx < blob_bottom_->count(); idx++) {
      filer >> val;
      blob_bottom_->mutable_cpu_data()[idx] = val;
    }
    filer.close();


    // ###############################################################
    // SetUp
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    // 
    EXPECT_EQ(this->blob_top_a_->num(), ims_per_batch);
    EXPECT_EQ(this->blob_top_a_->channels(), channels);
    EXPECT_EQ(this->blob_top_a_->height(), height);
    EXPECT_EQ(this->blob_top_a_->width(), width);
    // 
    EXPECT_EQ(this->blob_top_b_->num(), num - ims_per_batch);
    EXPECT_EQ(this->blob_top_b_->channels(), channels);
    EXPECT_EQ(this->blob_top_b_->height(), height);
    EXPECT_EQ(this->blob_top_b_->width(), width);


    // ###############################################################
    // Forward
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);


    // ###############################################################
    // Check blob_top_a_
    std::string top_data_output_3_filename_a = "top_data_output_3_a.log";
    std::string top_data_output_3_filepath_a = 
        test_file_dir + test_file_sub_dir + top_data_output_3_filename_a;
    // 
    std::ifstream filer1(top_data_output_3_filepath_a.c_str());
    CHECK(filer1);

    for(int idx = 0; idx < blob_top_a_->count(); idx++) {
      filer1 >> val;
      EXPECT_EQ(blob_top_a_->cpu_data()[idx], val);
    }
    filer1.close();

    // Check blob_top_b_
    std::string top_data_output_3_filename_b = "top_data_output_3_b.log";
    std::string top_data_output_3_filepath_b = 
        test_file_dir + test_file_sub_dir + top_data_output_3_filename_b;
    // 
    std::ifstream filer2(top_data_output_3_filepath_b.c_str());
    CHECK(filer2);

    for(int idx = 0; idx < blob_top_b_->count(); idx++) {
      filer2 >> val;
      EXPECT_EQ(blob_top_b_->cpu_data()[idx], val);
    }
    filer2.close();


    // ###############################################################
    // init diff values
    std::string top_diff_input_3_filename_a = "top_diff_input_3_a.log";
    std::string top_diff_input_3_filepath_a = 
        test_file_dir + test_file_sub_dir + top_diff_input_3_filename_a;

    std::ifstream filer3(top_diff_input_3_filepath_a.c_str());
    CHECK(filer3);

    for(int idx = 0; idx < blob_top_a_->count(); idx++) {
      filer3 >> val;
      blob_top_a_->mutable_cpu_diff()[idx] = val;
    }
    filer3.close();
    //
    std::string top_diff_input_3_filename_b = "top_diff_input_3_b.log";
    std::string top_diff_input_3_filepath_b = 
        test_file_dir + test_file_sub_dir + top_diff_input_3_filename_b;

    std::ifstream filer4(top_diff_input_3_filepath_b.c_str());
    CHECK(filer4);

    for(int idx = 0; idx < blob_top_b_->count(); idx++) {
      filer4 >> val;
      blob_top_b_->mutable_cpu_diff()[idx] = val;
    }
    filer4.close();




    // ###############################################################
    // Backward
    layer.Backward(blob_top_vec_, propagate_down_, blob_bottom_vec_);


    // 
    // Check
    std::string bottom_diff_input_3_filename = "bottom_diff_output_3.log";
    std::string bottom_diff_input_3_filepath = 
        test_file_dir + test_file_sub_dir + bottom_diff_input_3_filename;

    std::ifstream filer5(bottom_diff_input_3_filepath.c_str());
    CHECK(filer5);

    for(int idx = 0; idx < blob_bottom_->count(); idx++) {
      filer5 >> val;
      blob_bottom_->mutable_cpu_diff()[idx] = val;
    }
    filer5.close();
  }
};



// ################################################################



TYPED_TEST_CASE(ExtractPrimaryLayerTest2, TestDtypesAndDevices);



// layer_param.mutable_extract_primary_param()->set_axis(0);
TYPED_TEST(ExtractPrimaryLayerTest2, TestSetup) {
  this->TestSetup1();
  this->TestSetup2();
}


// where ims_per_batch is 2
TYPED_TEST(ExtractPrimaryLayerTest2, Test) {
  this->TestForwardBackward2();
  this->TestForwardBackward3();
}


}  // namespace caffe