#include <fcntl.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdint.h>

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/convert_img_blob.hpp"

namespace caffe {

cv::Mat ImageRead(const string& filename, const bool is_color) {
  int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR : CV_LOAD_IMAGE_GRAYSCALE);
  cv::Mat cv_img = cv::imread(filename, cv_read_flag);
  if (!cv_img.data) {
    LOG(ERROR) << "Could not open or find file " << filename;
  }
  return cv_img;
}

}  // namespace caffe
