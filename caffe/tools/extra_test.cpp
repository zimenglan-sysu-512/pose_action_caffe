#include <stdio.h>  // for snprintf
#include <string>
#include <vector>

#include "boost/algorithm/string.hpp"
#include "google/protobuf/text_format.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

using caffe::Blob;
using caffe::Caffe;
using caffe::Datum;
using caffe::Net;
using boost::shared_ptr;
using std::string;
namespace db = caffe::db;

template<typename Dtype>
void extra_test(int argc, char** argv);

int main(int argc, char** argv) {
   // extra_test<float>(argc, argv);
   LOG(INFO) << "ddk...";
   std::cout << "ddk..." << std::endl;
   Blob<float> interval;
   int num = 10;
   int channels = 5;
   int height = 1;
   int width = 1;
   interval.Reshape(num, channels, height, width);
   int count = interval.count();
   float mod = 5;
   caffe::caffe_cpu_intervals(count, mod, interval.mutable_cpu_data());

   const float* data = interval.cpu_data();
   int idx = 0;
   for(int n = 0; n < num; n++) {
    for(int c = 0; c < channels; c++) {
      for(int h = 0; h < height; h++) {
        for(int w = 0; w < width; w++) {
          std::cout << data[idx++] << " ";
        }
      }
    }
    std::cout << std::endl;
   }
   return 0;
}

template<typename Dtype>
void extra_test(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  LOG(INFO) << "extra test...";
}