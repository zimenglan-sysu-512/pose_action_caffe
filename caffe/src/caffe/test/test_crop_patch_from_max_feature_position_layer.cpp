#include <cstring>
#include <string>
#include <vector>

#include "google/protobuf/text_format.h"
#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/net.hpp"

#include "caffe/zhu_face_layers.hpp"
#include "caffe/util/util_coords.hpp"
#include "caffe/util/util_img.hpp"
#include "caffe/util/util_pre_define.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class CropPatchFromMaxFeaturePositionLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  CropPatchFromMaxFeaturePositionLayerTest() {}
  virtual ~CropPatchFromMaxFeaturePositionLayerTest() { }
};

TYPED_TEST_CASE(CropPatchFromMaxFeaturePositionLayerTest, TestDtypesAndDevices);

TYPED_TEST(CropPatchFromMaxFeaturePositionLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;

  const string list = "/home/black/caffe/caffe/caffe/src/caffe/test/test_data/zhu.face.test.data/crop_patch_max_fea_pos/prototxt.list";
  std::ifstream input_file(list.c_str());
  string filename;
  while(input_file >> filename) {
    Net<Dtype> net(filename, caffe::TEST);
    CropPatchFromMaxFeaturePositionLayer<Dtype>* layer =
    		dynamic_cast<CropPatchFromMaxFeaturePositionLayer<Dtype>* >(net.layers().back().get());
    EXPECT_TRUE(layer);

    std::ifstream result_file((filename + ".output").c_str());
    for (int i = 0; i < net.top_vecs().back().size(); ++i) {
    	int n, c, h, w;
    	result_file >> n >> c >> h >> w;
    	EXPECT_EQ(n, net.top_vecs().back()[i]->num());
    	EXPECT_EQ(c, net.top_vecs().back()[i]->channels());
    	EXPECT_EQ(h, net.top_vecs().back()[i]->height());
    	EXPECT_EQ(w, net.top_vecs().back()[i]->width());
    }

    result_file.close();
 }
  input_file.close();
}

TYPED_TEST(CropPatchFromMaxFeaturePositionLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;

  {
		const string net_filename = "/home/black/caffe/caffe/caffe/src/caffe/test/test_data/zhu.face.test.data/crop_patch_max_fea_pos/test.prototxt";
		Net<Dtype> net(net_filename, caffe::TEST);

		const vector<Blob<Dtype>*>& input_blobs = net.bottom_vecs().back();
		const char* input_filename = "/home/black/caffe/caffe/caffe/src/caffe/test/test_data/zhu.face.test.data/crop_patch_max_fea_pos/crop.input";
		std::ifstream input(input_filename);
		for (int i = 0; i < input_blobs[2]->count(); ++i) {
			input >> input_blobs[2]->mutable_cpu_data()[i];
		}

		CropPatchFromMaxFeaturePositionLayer<Dtype>* layer =
				dynamic_cast<CropPatchFromMaxFeaturePositionLayer<Dtype>* >(net.layers().back().get());
		EXPECT_TRUE(layer);

		const vector<Blob<Dtype>* >& output_blob = net.output_blobs();

		const char* output_filename = "/home/black/caffe/caffe/caffe/src/caffe/test/test_data/zhu.face.test.data/crop_patch_max_fea_pos/crop.output";
		std::ifstream output(output_filename);
		const char* coord_map_filename = "/home/black/caffe/caffe/caffe/src/caffe/test/test_data/zhu.face.test.data/crop_patch_max_fea_pos/crop.coord_map";
		std::ifstream coord_map_file(coord_map_filename);

		while (input >> input_blobs[1]->mutable_cpu_data()[0]) {
			for (int i = 1; i < input_blobs[1]->count(); ++i) {
				input >> input_blobs[1]->mutable_cpu_data()[i];
			}
			layer->Forward(input_blobs, output_blob);

//			const char* test_out_file = "/home/black/caffe/caffe/caffe/src/caffe/test/test_data/zhu.face.test.data/crop_patch_max_fea_pos/crop.input.output";
//			std::ofstream test_out(test_out_file);
//			for (int i = 0; i < output_blob.size(); ++i) {
//				for (int n = 0; n < output_blob[i]->num(); ++n) {
//					for (int c = 0; c < output_blob[i]->channels(); ++c) {
//						for (int h = 0; h < output_blob[i]->height(); ++h) {
//							for (int w = 0; w < output_blob[i]->width(); ++w) {
//								test_out << output_blob[i]->data_at(n, c, h, w) << " ";
//							}
//							test_out <<  std::endl;
//						}
//						test_out <<  std::endl;
//					}
//					test_out <<  std::endl;
//				}
//				test_out <<  std::endl;
//			}

			Dtype value;
			for (int i = 0; i < output_blob.size(); ++i) {
				for (int j = 0; j < output_blob[i]->count(); ++j) {
					output >> value;
					EXPECT_NEAR(value, output_blob[i]->cpu_data()[j], 1e-6) << j;
				}
			}

			vector<pair<Dtype, Dtype> > coefs = layer->coord_map().coefs();
			Dtype a, b;
			for (int i = 0; i < coefs.size(); ++i) {
				coord_map_file >> a >> b;
				EXPECT_NEAR(coefs[i].first, a, 1e-6);
				EXPECT_NEAR(coefs[i].second, b, 1e-6);
			}
		}
		input.close();
		output.close();
		coord_map_file.close();
  }

  {
 		const string net_filename = "/home/black/caffe/caffe/caffe/src/caffe/test/test_data/zhu.face.test.data/crop_patch_max_fea_pos/test4.prototxt";
 		Net<Dtype> net(net_filename, caffe::TEST);

 		const vector<Blob<Dtype>*>& input_blobs = net.bottom_vecs().back();
 		const char* input_filename = "/home/black/caffe/caffe/caffe/src/caffe/test/test_data/zhu.face.test.data/crop_patch_max_fea_pos/crop4.input";
 		std::ifstream input(input_filename);
 		for (int i = 0; i < input_blobs[2]->count(); ++i) {
 			input >> input_blobs[2]->mutable_cpu_data()[i];
 		}

 		CropPatchFromMaxFeaturePositionLayer<Dtype>* layer =
 				dynamic_cast<CropPatchFromMaxFeaturePositionLayer<Dtype>* >(net.layers().back().get());
 		EXPECT_TRUE(layer);

 		const vector<Blob<Dtype>* >& output_blob = net.output_blobs();

 		const char* output_filename = "/home/black/caffe/caffe/caffe/src/caffe/test/test_data/zhu.face.test.data/crop_patch_max_fea_pos/crop4.output";
 		std::ifstream output(output_filename);
		const char* coord_map_filename = "/home/black/caffe/caffe/caffe/src/caffe/test/test_data/zhu.face.test.data/crop_patch_max_fea_pos/crop4.coord_map";
		std::ifstream coord_map_file(coord_map_filename);

 		while (input >> input_blobs[1]->mutable_cpu_data()[0]) {
 			for (int i = 1; i < input_blobs[1]->count(); ++i) {
 				input >> input_blobs[1]->mutable_cpu_data()[i];
 			}
 			layer->Forward(input_blobs, output_blob);

// 			const char* test_out_file = "/home/black/caffe/caffe/caffe/src/caffe/test/test_data/zhu.face.test.data/crop_patch_max_fea_pos/crop4.input.output";
// 			std::ofstream test_out(test_out_file);
// 			for (int i = 0; i < output_blob.size(); ++i) {
// 				for (int n = 0; n < output_blob[i]->num(); ++n) {
// 					for (int c = 0; c < output_blob[i]->channels(); ++c) {
// 						for (int h = 0; h < output_blob[i]->height(); ++h) {
// 							for (int w = 0; w < output_blob[i]->width(); ++w) {
// 								test_out << output_blob[i]->data_at(n, c, h, w) << " ";
// 							}
// 							test_out <<  std::endl;
// 						}
// 						test_out <<  std::endl;
// 					}
// 					test_out <<  std::endl;
// 				}
// 				test_out <<  std::endl;
// 			}

 			Dtype value;
 			for (int i = 0; i < output_blob.size(); ++i) {
 				for (int j = 0; j < output_blob[i]->count(); ++j) {
 					output >> value;
 					EXPECT_NEAR(value, output_blob[i]->cpu_data()[j], 1e-6) << j;
 				}
 				vector<pair<Dtype, Dtype> > coefs = layer->coord_map().coefs();
 				Dtype a, b;
 				for (int i = 0; i < coefs.size(); ++i) {
 					coord_map_file >> a >> b;
 					EXPECT_NEAR(coefs[i].first, a, 1e-6);
 					EXPECT_NEAR(coefs[i].second, b, 1e-6);
 				}
 			}
 			input.close();
 			output.close();
 			coord_map_file.close();
 		}
  }

	{
		const string net_filename = "/home/black/caffe/caffe/caffe/src/caffe/test/test_data/zhu.face.test.data/crop_patch_max_fea_pos/test5.prototxt";
		Net<Dtype> net(net_filename, caffe::TEST);

		const vector<Blob<Dtype>*>& input_blobs = net.bottom_vecs().back();
		const char* input_filename = "/home/black/caffe/caffe/caffe/src/caffe/test/test_data/zhu.face.test.data/crop_patch_max_fea_pos/crop5.input";
		std::ifstream input(input_filename);
		for (int j = 2; j < input_blobs.size(); ++j) {
			for (int i = 0; i < input_blobs[j]->count(); ++i) {
				input >> input_blobs[j]->mutable_cpu_data()[i];
			}
		}

		CropPatchFromMaxFeaturePositionLayer<Dtype>* layer =
				dynamic_cast<CropPatchFromMaxFeaturePositionLayer<Dtype>* >(net.layers().back().get());
		EXPECT_TRUE(layer);

		const vector<Blob<Dtype>* >& output_blob = net.output_blobs();

		const char* output_filename = "/home/black/caffe/caffe/caffe/src/caffe/test/test_data/zhu.face.test.data/crop_patch_max_fea_pos/crop5.output";
		std::ifstream output(output_filename);
		const char* coord_map_filename = "/home/black/caffe/caffe/caffe/src/caffe/test/test_data/zhu.face.test.data/crop_patch_max_fea_pos/crop5.coord_map";
		std::ifstream coord_map_file(coord_map_filename);

		while (input >> input_blobs[1]->mutable_cpu_data()[0]) {
			for (int i = 1; i < input_blobs[1]->count(); ++i) {
				input >> input_blobs[1]->mutable_cpu_data()[i];
			}
			layer->Forward(input_blobs, output_blob);

// 			const char* test_out_file = "/home/black/caffe/caffe/caffe/src/caffe/test/test_data/zhu.face.test.data/crop_patch_max_fea_pos/crop5.input.output";
// 			std::ofstream test_out(test_out_file);
// 			for (int i = 0; i < output_blob.size(); ++i) {
// 				for (int n = 0; n < output_blob[i]->num(); ++n) {
// 					for (int c = 0; c < output_blob[i]->channels(); ++c) {
// 						for (int h = 0; h < output_blob[i]->height(); ++h) {
// 							for (int w = 0; w < output_blob[i]->width(); ++w) {
// 								test_out << output_blob[i]->data_at(n, c, h, w) << " ";
// 							}
// 							test_out <<  std::endl;
// 						}
// 						test_out <<  std::endl;
// 					}
// 					test_out <<  std::endl;
// 				}
// 				test_out <<  std::endl;
// 			}

			Dtype value;
			for (int i = 0; i < output_blob.size(); ++i) {
				for (int j = 0; j < output_blob[i]->count(); ++j) {
					output >> value;
					EXPECT_NEAR(value, output_blob[i]->cpu_data()[j], 1e-6) << j;
				}
			}
			vector<pair<Dtype, Dtype> > coefs = layer->coord_map().coefs();
			Dtype a, b;
			for (int i = 0; i < coefs.size(); ++i) {
				coord_map_file >> a >> b;
				EXPECT_NEAR(coefs[i].first, a, 1e-6);
				EXPECT_NEAR(coefs[i].second, b, 1e-6);
			}
		}
		input.close();
		output.close();
		coord_map_file.close();
   }
}

TYPED_TEST(CropPatchFromMaxFeaturePositionLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;

  const string list = "/home/black/caffe/caffe/caffe/src/caffe/test/test_data/zhu.face.test.data/crop_patch_max_fea_pos/prototxt.list";
  std::ifstream input_file(list.c_str());
  string filename;
  while(input_file >> filename) {
    Net<Dtype> net(filename, caffe::TEST);
    CropPatchFromMaxFeaturePositionLayer<Dtype>* layer =
    		dynamic_cast<CropPatchFromMaxFeaturePositionLayer<Dtype>* >(net.layers().back().get());
    EXPECT_TRUE(layer);

		GradientChecker<Dtype> checker(1e-2, 1e-2);
		for (int i = 2; i < net.bottom_vecs().back().size(); ++i) {
			checker.CheckGradientExhaustive(layer, net.bottom_vecs().back(), net.top_vecs().back(), i);
		}
  }
  input_file.close();
}

}  // namespace caffe
