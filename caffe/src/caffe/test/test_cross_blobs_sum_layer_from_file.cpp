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
class CrossBlobsSumLayerTest2 : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  CrossBlobsSumLayerTest2()
      : blob_bottom_p_(new Blob<Dtype>(6, 2, 3, 2)),
        blob_bottom_s_(new Blob<Dtype>(6, 2, 3, 2)),
        blob_top_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_p_);
    filler.Fill(this->blob_bottom_s_);
    blob_bottom_vec_.push_back(blob_bottom_p_);
    blob_bottom_vec_.push_back(blob_bottom_s_);

    blob_top_vec_.push_back(blob_top_);

    propagate_down_.push_back(true);
  }
  virtual ~CrossBlobsSumLayerTest2() {
    delete blob_bottom_p_;
    delete blob_bottom_s_;
    delete blob_top_;
  }
  Blob<Dtype>* const blob_bottom_p_;
  Blob<Dtype>* const blob_bottom_s_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
  vector<bool> propagate_down_;

  // 
  void TestSetup() {
    typedef typename TypeParam::Dtype Dtype;
    LayerParameter layer_param;
    const int num_batch = 6;
    const int channels = 2;
    const int height = 3;
    const int width = 2;
    CrossBlobsSumLayer<Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

    EXPECT_EQ(this->blob_top_->num(), num_batch);
    EXPECT_EQ(this->blob_top_->channels(), channels);
    EXPECT_EQ(this->blob_top_->height(), height);
    EXPECT_EQ(this->blob_top_->width(), width);
  }

  


  // where num_batch is 2
  void TestForwardBackward()  {
    typedef typename TypeParam::Dtype Dtype;
    LayerParameter layer_param;
    const int num_batch = 6;
    const int channels = 2;
    const int height = 3;
    const int width = 2;
    CrossBlobsSumLayer<Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

    EXPECT_EQ(this->blob_top_->num(), num_batch);
    EXPECT_EQ(this->blob_top_->channels(), channels);
    EXPECT_EQ(this->blob_top_->height(), height);
    EXPECT_EQ(this->blob_top_->width(), width);


    // ###############################################################
    // init data values
    std::string test_file_dir = 
        "/home/black/caffe/fast-rcnn-action/caffe/src/caffe/test/";
    std::string test_file_sub_dir = "test_data/cross_blobs_sum/";

    // primary
    std::string bottom_data_input_filename_p = "bottom_data_input_p.log";
    std::string bottom_data_input_filepath_p = 
        test_file_dir + test_file_sub_dir + bottom_data_input_filename_p;
    // 
    std::ifstream filer_p(bottom_data_input_filepath_p.c_str());
    CHECK(filer_p);
    // 
    int val;
    for(int idx = 0; idx < blob_bottom_p_->count(); idx++) {
      filer_p >> val;
      blob_bottom_p_->mutable_cpu_data()[idx] = val;
    }
    filer_p.close();
    //
    // secondary
    std::string bottom_data_input_filename_s = "bottom_data_input_s.log";
    std::string bottom_data_input_filepath_s = 
        test_file_dir + test_file_sub_dir + bottom_data_input_filename_s;
    // 
    std::ifstream filer_s(bottom_data_input_filepath_s.c_str());
    CHECK(filer_s);
    // 
    for(int idx = 0; idx < blob_bottom_s_->count(); idx++) {
      filer_s >> val;
      blob_bottom_s_->mutable_cpu_data()[idx] = val;
    }
    filer_s.close();


    // ###############################################################
    // SetUp
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    // 
    // Forward
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);


    // ###############################################################
    // Check blob_top_
    std::string top_data_output_filename = "top_data_output.log";
    std::string top_data_output_filepath = 
        test_file_dir + test_file_sub_dir + top_data_output_filename;
    // 
    std::ifstream filer1(top_data_output_filepath.c_str());
    CHECK(filer1);

    for(int idx = 0; idx < blob_top_->count(); idx++) {
      filer1 >> val;
      EXPECT_EQ(blob_top_->cpu_data()[idx], val);
    }
    filer1.close();

    

    // ###############################################################
    // init diff values
    std::string top_diff_input_filename_a = "top_diff_input.log";
    std::string top_diff_input_filepath_a = 
        test_file_dir + test_file_sub_dir + top_diff_input_filename_a;

    std::ifstream filer2(top_diff_input_filepath_a.c_str());
    CHECK(filer2);

    for(int idx = 0; idx < blob_top_->count(); idx++) {
      filer2 >> val;
      blob_top_->mutable_cpu_diff()[idx] = val;
    }
    filer2.close();


    // ###############################################################
    // Backward
    layer.Backward(blob_top_vec_, propagate_down_, blob_bottom_vec_);


    // 
    // Check
    // primary
    std::string bottom_diff_input_filename_p = "bottom_diff_output_p.log";
    std::string bottom_diff_input_filepath_p = 
        test_file_dir + test_file_sub_dir + bottom_diff_input_filename_p;

    std::ifstream filer_p2(bottom_diff_input_filepath_p.c_str());
    CHECK(filer_p2);

    for(int idx = 0; idx < blob_bottom_p_->count(); idx++) {
      filer_p2 >> val;
      blob_bottom_p_->mutable_cpu_diff()[idx] = val;
    }
    filer_p2.close();
    // secondary
    std::string bottom_diff_input_filename_s = "bottom_diff_output_s.log";
    std::string bottom_diff_input_filepath_s = 
        test_file_dir + test_file_sub_dir + bottom_diff_input_filename_s;

    std::ifstream filer_s2(bottom_diff_input_filepath_s.c_str());
    CHECK(filer_s2);

    for(int idx = 0; idx < blob_bottom_s_->count(); idx++) {
      filer_s2 >> val;
      blob_bottom_s_->mutable_cpu_diff()[idx] = val;
    }
    filer_s2.close();
  }
};



// ################################################################



TYPED_TEST_CASE(CrossBlobsSumLayerTest2, TestDtypesAndDevices);



// layer_param.mutable_extract_primary_param()->set_axis(0);
TYPED_TEST(CrossBlobsSumLayerTest2, TestSetup) {
  this->TestSetup();
}


// where num_batch is 2
TYPED_TEST(CrossBlobsSumLayerTest2, Test) {
  this->TestForwardBackward();
}


}  // namespace caffe