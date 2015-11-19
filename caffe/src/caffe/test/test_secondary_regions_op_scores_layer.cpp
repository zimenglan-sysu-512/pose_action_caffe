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
class SecondaryRegionsOpScoresLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  SecondaryRegionsOpScoresLayerTest()
      : blob_bottom_(new Blob<Dtype>()),
        blob_top_(new Blob<Dtype>()),
        blob_top_sr_inds_(new Blob<Dtype>()),
        blob_top_mask_(new Blob<Dtype>()) {}
  // #####################################################################
  // 
  virtual void SetUp() {
    Caffe::set_random_seed(1701);
    blob_bottom_->Reshape(6, 3, 4, 4);
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    // 
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
    // 
    propagate_down_.push_back(true);
  }



  // #####################################################################
  // 
  virtual ~SecondaryRegionsOpScoresLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
    delete blob_top_sr_inds_;
    delete blob_top_mask_;
  }


  // #####################################################################
  // 
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  Blob<Dtype>* const blob_top_sr_inds_;
  Blob<Dtype>* const blob_top_mask_;

  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
  vector<bool> propagate_down_;



  // #####################################################################
  // SetUp (MAX) -- n_secondary_regions is 2
  // bottom: (6, 3, 4, 4)
  // top_0: (3, 3, 4, 4)
  // top_1: (3, 3, 4, 4)
  void TestForwardBackward_max_2() {
    LayerParameter layer_param;
    SecondaryRegionsOpScoresParameter* secondary_regions_op_scores_param = 
        layer_param.mutable_secondary_regions_op_scores_param();

    const int n_secondary_regions = 2;
    secondary_regions_op_scores_param->set_n_secondary_regions(n_secondary_regions);

    const int num = 6;
    const int channels = 3;
    const int height = 2;
    const int width = 2;
    blob_bottom_->Reshape(num, channels, height, width);

    // 
    std::string test_file_dir = "/home/black/caffe/fast-rcnn-action/caffe-fast-rcnn/src/caffe/test/";
    std::string test_file_sub_dir = "test_data/secondary_regions/max/";
    std::string bottom_data_input_filename = "bottom_data_input.log";
    std::string bottom_data_input_filepath = 
        test_file_dir + test_file_sub_dir + bottom_data_input_filename;

    std::ifstream filer(bottom_data_input_filepath.c_str());
    CHECK(filer);

    int val;
    for(int idx = 0; idx < blob_bottom_->count(); idx++) {
      filer >> val;
      blob_bottom_->mutable_cpu_data()[idx] = val;
    }
    filer.close();

    // SetUp
    SecondaryRegionsOpScoresLayer<Dtype> layer(layer_param);
    layer.SetUp(blob_bottom_vec_, blob_top_vec_);

    const int top_num = num / n_secondary_regions;
    EXPECT_EQ(blob_top_->num(), top_num);
    EXPECT_EQ(blob_top_->channels(), channels);
    EXPECT_EQ(blob_top_->height(), height);
    EXPECT_EQ(blob_top_->width(), width);

    // sr_inds & max_op_idxs
    if (blob_top_vec_.size() > 1) {
      // sr_inds
      EXPECT_EQ(blob_top_sr_inds_->num(), top_num);
      EXPECT_EQ(blob_top_sr_inds_->channels(), channels);
      EXPECT_EQ(blob_top_sr_inds_->height(), height);
      EXPECT_EQ(blob_top_sr_inds_->width(), width);

      // max_op_idxs
      if(blob_top_vec_.size() > 2) {
        EXPECT_EQ(blob_top_mask_->num(), top_num);
        EXPECT_EQ(blob_top_mask_->channels(), channels);
        EXPECT_EQ(blob_top_mask_->height(), height);
        EXPECT_EQ(blob_top_mask_->width(), width);  
      }
    }

    // ##################################################################
    // Forward
    layer.Forward(blob_bottom_vec_, blob_top_vec_);
    
    std::string top_data_output_2_filename = "top_data_output_2.log";
    std::string top_data_output_2_filepath = 
        test_file_dir + test_file_sub_dir + top_data_output_2_filename;
    // 
    std::ifstream filer1(top_data_output_2_filepath.c_str());
    CHECK(filer1);

    for(int idx = 0; idx < blob_top_->count(); idx++) {
      filer1 >> val;
      EXPECT_EQ(blob_top_->cpu_data()[idx], val);
    }
    filer1.close();

    // sr_inds & max_op_idxs
    if (blob_top_vec_.size() > 1) {
      // sr_inds
      std::string top_data_output_2_sr_inds_filename = "top_data_output_2_sr_inds.log";
      std::string top_data_output_2_sr_inds_filepath = 
          test_file_dir + test_file_sub_dir + top_data_output_2_sr_inds_filename;

      std::ifstream filer_sr_inds_2(top_data_output_2_sr_inds_filepath.c_str());
      CHECK(filer_sr_inds_2);

      for(int idx = 0; idx < blob_top_sr_inds_->count(); idx++) {
        filer_sr_inds_2 >> val;
        EXPECT_EQ(blob_top_sr_inds_->cpu_data()[idx], val);
      }
      filer_sr_inds_2.close();

      // max_op_idxs
      if(blob_top_vec_.size() > 2) {
        std::string top_data_output_2_idx_filename = "top_data_output_2_idx.log";
        std::string top_data_output_2_idx_filepath = 
            test_file_dir + test_file_sub_dir + top_data_output_2_idx_filename;
        // 
        std::ifstream filer2(top_data_output_2_idx_filepath.c_str());
        CHECK(filer2);

        for(int idx = 0; idx < blob_top_mask_->count(); idx++) {
          filer2 >> val;
          EXPECT_EQ(blob_top_mask_->cpu_data()[idx], val);
        }
        filer2.close();
      }
    }

    // ##############################################################################
    // Backward
    std::string top_diff_input_2_filename = "top_diff_input_2.log";
    std::string top_diff_input_2_filepath = 
        test_file_dir + test_file_sub_dir + top_diff_input_2_filename;

    std::ifstream filer3(top_diff_input_2_filepath.c_str());
    CHECK(filer3);

    for(int idx = 0; idx < blob_top_->count(); idx++) {
      filer3 >> val;
      blob_top_->mutable_cpu_diff()[idx] = val;
    }
    filer3.close();

    // Backward
    layer.Backward(blob_top_vec_, propagate_down_, blob_bottom_vec_);

    std::string bottom_diff_output_2_filename = "bottom_diff_output_2.log";
    std::string bottom_diff_output_2_filepath = 
        test_file_dir + test_file_sub_dir + bottom_diff_output_2_filename;

    std::ifstream filer4(bottom_diff_output_2_filepath.c_str());
    CHECK(filer4);

    for(int idx = 0; idx < blob_bottom_->count(); idx++) {
      filer4 >> val;
      EXPECT_EQ(blob_bottom_->cpu_diff()[idx], val);
    }
    filer4.close();
  }


  // #####################################################################
  // 
  // SetUp (MAX) -- n_secondary_regions is 3
  // bottom: (6, 3, 4, 4)
  // top_0: (2, 3, 4, 4)
  // top_1: (2, 3, 4, 4)
  void TestForwardBackward_max_3() {
    LayerParameter layer_param;
    SecondaryRegionsOpScoresParameter* secondary_regions_op_scores_param = 
        layer_param.mutable_secondary_regions_op_scores_param();

    const int n_secondary_regions = 3;
    secondary_regions_op_scores_param->set_n_secondary_regions(n_secondary_regions);

    const int num = 6;
    const int channels = 3;
    const int height = 2;
    const int width = 2;
    blob_bottom_->Reshape(num, channels, height, width);

    // 
    std::string test_file_dir = "/home/black/caffe/fast-rcnn-action/caffe-fast-rcnn/src/caffe/test/";
    std::string test_file_sub_dir = "test_data/secondary_regions/max/";
    std::string bottom_data_input_filename = "bottom_data_input.log";
    std::string bottom_data_input_filepath = 
        test_file_dir + test_file_sub_dir + bottom_data_input_filename;

    std::ifstream filer(bottom_data_input_filepath.c_str());
    CHECK(filer);

    int val;
    for(int idx = 0; idx < blob_bottom_->count(); idx++) {
      filer >> val;
      blob_bottom_->mutable_cpu_data()[idx] = val;
    }
    filer.close();

    // SetUp
    SecondaryRegionsOpScoresLayer<Dtype> layer(layer_param);
    layer.SetUp(blob_bottom_vec_, blob_top_vec_);

    const int top_num = num / n_secondary_regions;
    EXPECT_EQ(blob_top_->num(), top_num);
    EXPECT_EQ(blob_top_->channels(), channels);
    EXPECT_EQ(blob_top_->height(), height);
    EXPECT_EQ(blob_top_->width(), width);

    // sr_inds & max_op_idxs
    if (blob_top_vec_.size() > 1) {
      // sr_inds
      EXPECT_EQ(blob_top_sr_inds_->num(), top_num);
      EXPECT_EQ(blob_top_sr_inds_->channels(), channels);
      EXPECT_EQ(blob_top_sr_inds_->height(), height);
      EXPECT_EQ(blob_top_sr_inds_->width(), width);

      // max_op_idxs
      if(blob_top_vec_.size() > 2) {
        EXPECT_EQ(blob_top_mask_->num(), top_num);
        EXPECT_EQ(blob_top_mask_->channels(), channels);
        EXPECT_EQ(blob_top_mask_->height(), height);
        EXPECT_EQ(blob_top_mask_->width(), width);  
      }
    }

    // Forward
    layer.Forward(blob_bottom_vec_, blob_top_vec_);
    
    std::string top_data_output_3_filename = "top_data_output_3.log";
    std::string top_data_output_3_filepath = 
        test_file_dir + test_file_sub_dir + top_data_output_3_filename;
    // 
    std::ifstream filer1(top_data_output_3_filepath.c_str());
    CHECK(filer1);

    for(int idx = 0; idx < blob_top_->count(); idx++) {
      filer1 >> val;
      EXPECT_EQ(blob_top_->cpu_data()[idx], val);
    }
    filer1.close();

    
    // sr_inds & max_op_idxs
    if (blob_top_vec_.size() > 1) {
      // sr_inds
      std::string top_data_output_3_sr_inds_filename = "top_data_output_3_sr_inds.log";
      std::string top_data_output_3_sr_inds_filepath = 
          test_file_dir + test_file_sub_dir + top_data_output_3_sr_inds_filename;

      std::ifstream filer_sr_inds_3(top_data_output_3_sr_inds_filepath.c_str());
      CHECK(filer_sr_inds_3);

      for(int idx = 0; idx < blob_top_sr_inds_->count(); idx++) {
        filer_sr_inds_3 >> val;
        EXPECT_EQ(blob_top_sr_inds_->cpu_data()[idx], val);
      }
      filer_sr_inds_3.close();

      // max_op_idxs
      if(blob_top_vec_.size() > 2) {
        std::string top_data_output_3_idx_filename = "top_data_output_3_idx.log";
        std::string top_data_output_3_idx_filepath = 
            test_file_dir + test_file_sub_dir + top_data_output_3_idx_filename;
        // 
        std::ifstream filer2(top_data_output_3_idx_filepath.c_str());
        CHECK(filer2);

        for(int idx = 0; idx < blob_top_mask_->count(); idx++) {
          filer2 >> val;
          EXPECT_EQ(blob_top_mask_->cpu_data()[idx], val);
        }
        filer2.close();
      }
    }

    // ###########################################################################
    // Backward
    std::string top_diff_input_3_filename = "top_diff_input_3.log";
    std::string top_diff_input_3_filepath = 
        test_file_dir + test_file_sub_dir + top_diff_input_3_filename;

    std::ifstream filer3(top_diff_input_3_filepath.c_str());
    CHECK(filer3);

    for(int idx = 0; idx < blob_top_->count(); idx++) {
      filer3 >> val;
      blob_top_->mutable_cpu_diff()[idx] = val;
    }
    filer3.close();

    // Backward
    layer.Backward(blob_top_vec_, propagate_down_, blob_bottom_vec_);

    std::string bottom_diff_output_3_filename = "bottom_diff_output_3.log";
    std::string bottom_diff_output_3_filepath = 
        test_file_dir + test_file_sub_dir + bottom_diff_output_3_filename;

    std::ifstream filer4(bottom_diff_output_3_filepath.c_str());
    CHECK(filer4);

    for(int idx = 0; idx < blob_bottom_->count(); idx++) {
      filer4 >> val;
      EXPECT_EQ(blob_bottom_->cpu_diff()[idx], val);
    }
    filer4.close();
  }


  // #####################################################################
  // 
  // SetUp (MAX) -- n_secondary_regions is 6
  // bottom: (6, 3, 4, 4)
  // top_0: (1, 3, 4, 4)
  // top_1: (1, 3, 4, 4)
  void TestForwardBackward_max_6() {
    LayerParameter layer_param;
    SecondaryRegionsOpScoresParameter* secondary_regions_op_scores_param = 
        layer_param.mutable_secondary_regions_op_scores_param();

    const int n_secondary_regions = 6;
    secondary_regions_op_scores_param->set_n_secondary_regions(n_secondary_regions);

    const int num = 6;
    const int channels = 3;
    const int height = 2;
    const int width = 2;
    blob_bottom_->Reshape(num, channels, height, width);

    // 
    std::string test_file_dir = "/home/black/caffe/fast-rcnn-action/caffe-fast-rcnn/src/caffe/test/";
    std::string test_file_sub_dir = "test_data/secondary_regions/max/";
    std::string bottom_data_input_filename = "bottom_data_input.log";
    std::string bottom_data_input_filepath = 
        test_file_dir + test_file_sub_dir + bottom_data_input_filename;

    std::ifstream filer(bottom_data_input_filepath.c_str());
    CHECK(filer);

    int val;
    for(int idx = 0; idx < blob_bottom_->count(); idx++) {
      filer >> val;
      blob_bottom_->mutable_cpu_data()[idx] = val;
    }
    filer.close();

    // SetUp
    SecondaryRegionsOpScoresLayer<Dtype> layer(layer_param);
    layer.SetUp(blob_bottom_vec_, blob_top_vec_);

    const int top_num = num / n_secondary_regions;
    EXPECT_EQ(blob_top_->num(), top_num);
    EXPECT_EQ(blob_top_->channels(), channels);
    EXPECT_EQ(blob_top_->height(), height);
    EXPECT_EQ(blob_top_->width(), width);

    // sr_inds & max_op_idxs
    if (blob_top_vec_.size() > 1) {
      // sr_inds
      EXPECT_EQ(blob_top_sr_inds_->num(), top_num);
      EXPECT_EQ(blob_top_sr_inds_->channels(), channels);
      EXPECT_EQ(blob_top_sr_inds_->height(), height);
      EXPECT_EQ(blob_top_sr_inds_->width(), width);

      // max_op_idxs
      if(blob_top_vec_.size() > 2) {
        EXPECT_EQ(blob_top_mask_->num(), top_num);
        EXPECT_EQ(blob_top_mask_->channels(), channels);
        EXPECT_EQ(blob_top_mask_->height(), height);
        EXPECT_EQ(blob_top_mask_->width(), width);  
      }
    }

    // #################################################################################
    // Forward
    layer.Forward(blob_bottom_vec_, blob_top_vec_);
    
    std::string top_data_output_6_filename = "top_data_output_6.log";
    std::string top_data_output_6_filepath = 
        test_file_dir + test_file_sub_dir + top_data_output_6_filename;
    // 
    std::ifstream filer1(top_data_output_6_filepath.c_str());
    CHECK(filer1);

    for(int idx = 0; idx < blob_top_->count(); idx++) {
      filer1 >> val;
      EXPECT_EQ(blob_top_->cpu_data()[idx], val);
    }
    filer1.close();

    // sr_inds & max_op_idxs
    if (blob_top_vec_.size() > 1) {
      // sr_inds
      std::string top_data_output_6_sr_inds_filename = "top_data_output_6_sr_inds.log";
      std::string top_data_output_6_sr_inds_filepath = 
          test_file_dir + test_file_sub_dir + top_data_output_6_sr_inds_filename;

      std::ifstream filer_sr_inds_6(top_data_output_6_sr_inds_filepath.c_str());
      CHECK(filer_sr_inds_6);

      for(int idx = 0; idx < blob_top_sr_inds_->count(); idx++) {
        filer_sr_inds_6 >> val;
        EXPECT_EQ(blob_top_sr_inds_->cpu_data()[idx], val);
      }
      filer_sr_inds_6.close();

      // max_op_idxs
      if(blob_top_vec_.size() > 2) {
        std::string top_data_output_6_idx_filename = "top_data_output_6_idx.log";
        std::string top_data_output_6_idx_filepath = 
            test_file_dir + test_file_sub_dir + top_data_output_6_idx_filename;
        // 
        std::ifstream filer2(top_data_output_6_idx_filepath.c_str());
        CHECK(filer2);

        for(int idx = 0; idx < blob_top_mask_->count(); idx++) {
          filer2 >> val;
          EXPECT_EQ(blob_top_mask_->cpu_data()[idx], val);
        }
        filer2.close();
      }
    }

    // ##############################################################################
    // Backward
    std::string top_diff_input_6_filename = "top_diff_input_6.log";
    std::string top_diff_input_6_filepath = 
        test_file_dir + test_file_sub_dir + top_diff_input_6_filename;

    std::ifstream filer3(top_diff_input_6_filepath.c_str());
    CHECK(filer3);

    for(int idx = 0; idx < blob_top_->count(); idx++) {
      filer3 >> val;
      blob_top_->mutable_cpu_diff()[idx] = val;
    }
    filer3.close();

    // Backward
    layer.Backward(blob_top_vec_, propagate_down_, blob_bottom_vec_);

    std::string bottom_diff_output_6_filename = "bottom_diff_output_6.log";
    std::string bottom_diff_output_6_filepath = 
        test_file_dir + test_file_sub_dir + bottom_diff_output_6_filename;

    std::ifstream filer4(bottom_diff_output_6_filepath.c_str());
    CHECK(filer4);

    for(int idx = 0; idx < blob_bottom_->count(); idx++) {
      filer4 >> val;
      EXPECT_EQ(blob_bottom_->cpu_diff()[idx], val);
    }
    filer4.close();
  }



  /* ******************************************************************** */


  // #####################################################################
  // SetUp (SUM) -- n_secondary_regions is 2
  // bottom: (6, 3, 4, 4)
  // top: (3, 3, 4, 4)
  void TestForwardBackward_sum_2() {
    LayerParameter layer_param;
    SecondaryRegionsOpScoresParameter* secondary_regions_op_scores_param = 
        layer_param.mutable_secondary_regions_op_scores_param();

    const int n_secondary_regions = 2;
    secondary_regions_op_scores_param->set_n_secondary_regions(n_secondary_regions);
    secondary_regions_op_scores_param->set_op_scores(
        SecondaryRegionsOpScoresParameter_OpScoresMethod_SUM);

    const int num = 6;
    const int channels = 3;
    const int height = 2;
    const int width = 2;
    blob_bottom_->Reshape(num, channels, height, width);

    // 
    std::string test_file_dir = "/home/black/caffe/fast-rcnn-action/caffe-fast-rcnn/src/caffe/test/";
    std::string test_file_sub_dir = "test_data/secondary_regions/sum/";
    std::string bottom_data_input_filename = "bottom_data_input.log";
    std::string bottom_data_input_filepath = 
        test_file_dir + test_file_sub_dir + bottom_data_input_filename;

    std::ifstream filer(bottom_data_input_filepath.c_str());
    CHECK(filer);

    int val;
    for(int idx = 0; idx < blob_bottom_->count(); idx++) {
      filer >> val;
      blob_bottom_->mutable_cpu_data()[idx] = val;
    }
    filer.close();

    // SetUp
    SecondaryRegionsOpScoresLayer<Dtype> layer(layer_param);
    layer.SetUp(blob_bottom_vec_, blob_top_vec_);

    const int top_num = num / n_secondary_regions;
    EXPECT_EQ(blob_top_->num(), top_num);
    EXPECT_EQ(blob_top_->channels(), channels);
    EXPECT_EQ(blob_top_->height(), height);
    EXPECT_EQ(blob_top_->width(), width);

    // ##################################################################
    // Forward
    layer.Forward(blob_bottom_vec_, blob_top_vec_);
    
    std::string top_data_output_2_filename = "top_data_output_2.log";
    std::string top_data_output_2_filepath = 
        test_file_dir + test_file_sub_dir + top_data_output_2_filename;
    // 
    std::ifstream filer1(top_data_output_2_filepath.c_str());
    CHECK(filer1);

    for(int idx = 0; idx < blob_top_->count(); idx++) {
      filer1 >> val;
      EXPECT_EQ(blob_top_->cpu_data()[idx], val);
    }
    filer1.close();

    // #################################################################
    // Backward
    std::string top_diff_input_2_filename = "top_diff_input_2.log";
    std::string top_diff_input_2_filepath = 
        test_file_dir + test_file_sub_dir + top_diff_input_2_filename;

    std::ifstream filer3(top_diff_input_2_filepath.c_str());
    CHECK(filer3);

    for(int idx = 0; idx < blob_top_->count(); idx++) {
      filer3 >> val;
      blob_top_->mutable_cpu_diff()[idx] = val;
    }
    filer3.close();

    // Backward
    layer.Backward(blob_top_vec_, propagate_down_, blob_bottom_vec_);

    std::string bottom_diff_output_2_filename = "bottom_diff_output_2.log";
    std::string bottom_diff_output_2_filepath = 
        test_file_dir + test_file_sub_dir + bottom_diff_output_2_filename;

    std::ifstream filer4(bottom_diff_output_2_filepath.c_str());
    CHECK(filer4);

    for(int idx = 0; idx < blob_bottom_->count(); idx++) {
      filer4 >> val;
      EXPECT_EQ(blob_bottom_->cpu_diff()[idx], val);
    }
    filer4.close();
  }

  // #####################################################################
  // SetUp (SUM) -- n_secondary_regions is 2
  // bottom: (6, 3, 4, 4)
  // top: (3, 3, 4, 4)
  void TestForwardBackward_ave_2() {
    LayerParameter layer_param;
    SecondaryRegionsOpScoresParameter* secondary_regions_op_scores_param = 
        layer_param.mutable_secondary_regions_op_scores_param();

    const int n_secondary_regions = 2;
    secondary_regions_op_scores_param->set_n_secondary_regions(n_secondary_regions);
    secondary_regions_op_scores_param->set_op_scores(
        SecondaryRegionsOpScoresParameter_OpScoresMethod_AVE);

    const int num = 6;
    const int channels = 3;
    const int height = 2;
    const int width = 2;
    blob_bottom_->Reshape(num, channels, height, width);

    // 
    std::string test_file_dir = "/home/black/caffe/fast-rcnn-action/caffe-fast-rcnn/src/caffe/test/";
    std::string test_file_sub_dir = "test_data/secondary_regions/sum/";
    std::string bottom_data_input_filename = "bottom_data_input.log";
    std::string bottom_data_input_filepath = 
        test_file_dir + test_file_sub_dir + bottom_data_input_filename;

    std::ifstream filer(bottom_data_input_filepath.c_str());
    CHECK(filer);

    int val;
    for(int idx = 0; idx < blob_bottom_->count(); idx++) {
      filer >> val;
      blob_bottom_->mutable_cpu_data()[idx] = val;
    }
    filer.close();

    // SetUp
    SecondaryRegionsOpScoresLayer<Dtype> layer(layer_param);
    layer.SetUp(blob_bottom_vec_, blob_top_vec_);

    const int top_num = num / n_secondary_regions;
    EXPECT_EQ(blob_top_->num(), top_num);
    EXPECT_EQ(blob_top_->channels(), channels);
    EXPECT_EQ(blob_top_->height(), height);
    EXPECT_EQ(blob_top_->width(), width);

    // ##################################################################
    // Forward
    layer.Forward(blob_bottom_vec_, blob_top_vec_);
    
    std::string top_data_output_2_filename = "top_data_output_2_ave.log";
    std::string top_data_output_2_filepath = 
        test_file_dir + test_file_sub_dir + top_data_output_2_filename;
    // 
    std::ifstream filer1(top_data_output_2_filepath.c_str());
    CHECK(filer1);

    for(int idx = 0; idx < blob_top_->count(); idx++) {
      filer1 >> val;
      EXPECT_EQ(blob_top_->cpu_data()[idx], val);
    }
    filer1.close();

    // Backward
    std::string top_diff_input_2_filename = "top_diff_input_2.log";
    std::string top_diff_input_2_filepath = 
        test_file_dir + test_file_sub_dir + top_diff_input_2_filename;

    std::ifstream filer3(top_diff_input_2_filepath.c_str());
    CHECK(filer3);

    for(int idx = 0; idx < blob_top_->count(); idx++) {
      filer3 >> val;
      blob_top_->mutable_cpu_diff()[idx] = val;
    }
    filer3.close();

    // Backward
    layer.Backward(blob_top_vec_, propagate_down_, blob_bottom_vec_);

    std::string bottom_diff_output_2_filename = "bottom_diff_output_2.log";
    std::string bottom_diff_output_2_filepath = 
        test_file_dir + test_file_sub_dir + bottom_diff_output_2_filename;

    std::ifstream filer4(bottom_diff_output_2_filepath.c_str());
    CHECK(filer4);

    for(int idx = 0; idx < blob_bottom_->count(); idx++) {
      filer4 >> val;
      EXPECT_EQ(blob_bottom_->cpu_diff()[idx], val);
    }
    filer4.close();
  }

};




/* ******************************************************************** */




TYPED_TEST_CASE(SecondaryRegionsOpScoresLayerTest, TestDtypesAndDevices);




/* ******************************************************************** */



// MAX - without idxs
TYPED_TEST(SecondaryRegionsOpScoresLayerTest, TestForwardBackward_Max) {
  // n_secondary_regions: 2
  this->TestForwardBackward_max_2();
  // n_secondary_regions: 3
  this->TestForwardBackward_max_3();
  // n_secondary_regions: 6
  this->TestForwardBackward_max_6();
}

// MAX - with sr_inds
TYPED_TEST(SecondaryRegionsOpScoresLayerTest, TestForwardBackward_MaxTopMask) {
  this->blob_top_vec_.push_back(this->blob_top_sr_inds_);
  this->propagate_down_.push_back(true);

  // n_secondary_regions: 2
  this->TestForwardBackward_max_2();
  // n_secondary_regions: 3
  this->TestForwardBackward_max_3();
  // n_secondary_regions: 6
  this->TestForwardBackward_max_6();
}


// MAX - with sr_inds & max_op_idxs
TYPED_TEST(SecondaryRegionsOpScoresLayerTest, TestForwardBackward_MaxTopMask2) {
  this->blob_top_vec_.push_back(this->blob_top_sr_inds_);
  this->blob_top_vec_.push_back(this->blob_top_mask_);
  this->propagate_down_.push_back(true);
  this->propagate_down_.push_back(true);

  // n_secondary_regions: 2
  this->TestForwardBackward_max_2();
  // n_secondary_regions: 3
  this->TestForwardBackward_max_3();
  // n_secondary_regions: 6
  this->TestForwardBackward_max_6();
}



// #####################################################################


// SUM
TYPED_TEST(SecondaryRegionsOpScoresLayerTest, TestForwardBackward_Sum) {
  // n_secondary_regions: 2
  this->TestForwardBackward_sum_2();
}


// Ave
TYPED_TEST(SecondaryRegionsOpScoresLayerTest, TestForwardBackward_Ave) {
  // n_secondary_regions: 2
  this->TestForwardBackward_ave_2();
}



}  // namespace caffe