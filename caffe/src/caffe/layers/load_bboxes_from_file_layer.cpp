// CopyRight by Dengke Dong

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>
#include <map>
#include <boost/thread.hpp>
#include "boost/algorithm/string.hpp"

#include "caffe/pose_estimation_layers.hpp"
#include "caffe/util/pose_tool.hpp"
#include "caffe/global_variables.hpp"
#include "caffe/util/math_functions.hpp"

#define __LOAD_BBOX_FROM_FILE_LAYER_VISUAL__
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

namespace caffe {

template <typename Dtype>
void LoadBboxFromFileLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
  const LoadBboxFromFileParameter load_bbox_from_file_param = 
      this->layer_param_.load_bbox_from_file_param();
  CHECK(load_bbox_from_file_param.has_bbox_file());
  bbox_file_ = load_bbox_from_file_param.bbox_file();

  LOG(INFO) << "Opening file " << bbox_file_;
  std::ifstream filer(bbox_file_.c_str());
  CHECK(filer);

  int im_ind;
  string line;
  string objidx;
  string imgidx;

  // format: `imgidx objidx x1 y1 x2 y2` (each line)
  while (getline(filer, line)) {
    std::vector<std::string> info;
    boost::trim(line);
    boost::split(info, line, boost::is_any_of(" "));
    CHECK_GE(info.size(), 6);
    // imgidx & objidx 
    imgidx = info[0];
    objidx = info[1];
    im_ind = std::atoi(objidx.c_str());
    // bbox: x1, y1, x2, y2
    std::vector<float> bbox;
    for(int j = 0; j < 4; j++) {
      bbox.push_back(std::atof(info[j].c_str()));
    }
    // swap
    if(bbox[0] > bbox[2]) std::swap(bbox[0], bbox[2]);
    if(bbox[1] > bbox[3]) std::swap(bbox[1], bbox[3]);
    // set
    lines_[imgidx][im_ind] = bbox;
  }
  LOG(INFO) << "total images: " << lines_.size();
  LOG(INFO) << "loading bbox from `" << bbox_file_ << "` done";

  img_ext_         = load_bbox_from_file_param.img_ext();
  is_color_        = load_bbox_from_file_param.is_color();
  root_folder_     = load_bbox_from_file_param.root_folder();
  visual_path_     = load_bbox_from_file_param.visual_path();
  has_visual_path_ = load_bbox_from_file_param.has_root_folder()
      && load_bbox_from_file_param.has_visual_path() 
      && root_folder_.length() > 0 and visual_path_.length() > 1;
  if(has_visual_path_) {
    CreateDir(visual_path_.c_str(), 0);
    LOG(INFO) << "root_folder: " << root_folder_;
    LOG(INFO) << "visual_path: " << visual_path_;
  }
}

template <typename Dtype>
void LoadBboxFromFileLayer<Dtype>::Reshape( /*bottom[0]: aux_info*/
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
  const int num      = bottom[0]->num();
  const int channels = bottom[0]->channels();
  const int count    = bottom[0]->count();
  CHECK_EQ(channels, count / num);

  top[0]->Reshape(num, 4, 1, 1);
}

template <typename Dtype>
void LoadBboxFromFileLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
  int num      = top[0]->num();
  int count    = top[0]->count();

  const std::vector<std::string> objidxs = GlobalVars::objidxs();
  const std::vector<std::string> imgidxs = GlobalVars::imgidxs();
  CHECK_EQ(objidxs.size(), num);
  CHECK_EQ(imgidxs.size(), num);

  const Dtype* aux_info = bottom[0]->cpu_data();
  Dtype* data_blob      = top[0]->mutable_cpu_data();
  caffe_set(count, Dtype(0), data_blob);

  int objidx;
  Dtype im_ind;
  Dtype x1, y1, x2, y2;
  Dtype One  = Dtype(1);
  Dtype Zero = Dtype(0);
  Dtype im_width, im_height;
  Dtype im_scale, im_flipped;
  std::string imgidx;

   // (im_ind, width, height, im_scale, flippable)
  for(int n_idx = 0; n_idx < num; n_idx++) {
    const int aux_info_off = bottom[0]->offset(n_idx);
    const int bbox_off     = top[0]->offset(n_idx);
    im_ind     = aux_info[aux_info_off + 0];
    im_width   = aux_info[aux_info_off + 1];
    im_height  = aux_info[aux_info_off + 2];
    im_scale   = aux_info[aux_info_off + 3];
    im_flipped = aux_info[aux_info_off + 4];

    objidx = int(im_ind);
    CHECK_EQ(objidxs[n_idx], boost::to_string(objidx));
    imgidx = imgidxs[n_idx];
    const std::vector<float>& bbox = lines_[imgidx][objidx];
    CHECK_EQ(bbox.size(), 4);

    x1 = Dtype(bbox[0]);
    y1 = Dtype(bbox[1]);
    x2 = Dtype(bbox[2]);
    y2 = Dtype(bbox[3]);
    CHECK_GT(x1, Zero);
    CHECK_GT(y1, Zero);
    CHECK_LT(x2, im_width  - One);
    CHECK_LT(y2, im_height - One);

    // flip
    if(im_flipped) {
      x1 = im_width - x1 - One;
      x2 = im_width - x2 - One;
    }
    
    // rescale
    x1 *= im_scale;
    y1 *= im_scale;
    x2 *= im_scale;
    y2 *= im_scale;

    // set
    data_blob[bbox_off + 0] = x1;
    data_blob[bbox_off + 1] = y1;
    data_blob[bbox_off + 2] = x2;
    data_blob[bbox_off + 3] = y2;

    // visualization
    #ifdef __LOAD_BBOX_FROM_FILE_LAYER_VISUAL__
    {
      if(!has_visual_path_) {
        continue;
      }
      std::string im_path = root_folder_ + imgidx + img_ext_;
      int flag = (is_color_ ? CV_LOAD_IMAGE_COLOR : CV_LOAD_IMAGE_GRAYSCALE);
      cv::Mat im = cv::imread(im_path, flag);
      if (!im.data) {
        LOG(ERROR) << "1 Could not open or find file " << im_path;
        return;
      }
      // check whether match the size
      int width  = int(im_width); 
      int height = int(im_height);
      CHECK_EQ(width,  im.cols) << "does not match the width (size) of input image";
      CHECK_EQ(height, im.rows) << "does not match the height (size) of input image";

      cv::Mat im2;
      int width2  = int(im_width  * im_scale);
      int height2 = int(im_height * im_scale);
      cv::resize(im, im2, cv::Size(width2, height2));
      CHECK(im2.data) << "2 Could not open or find file " << im_path;

      if(im_flipped) {
        // >0: horizontal; <0: horizontal&vertical; =0: vertical
        const int flipCode = 1;
        cv::flip(im2, im2, flipCode);
      }
      const std::string im_path2 = visual_path_ + imgidx + img_ext_;
      LOG(INFO) << "visualized image path: " << im_path2;
      cv::rectangle(im2, cv::Point(int(x1), int(y1)), 
          cv::Point(int(x2), int(y2)), cv::Scalar(213, 10, 232), 2);
      cv::imwrite(im_path2, im2);

      // cv::imshow(imgidx, im2);
      // cv::waitKey(0);
      // cv::destroyAllWindows();
    }
    #endif
  }
}

template <typename Dtype>
void LoadBboxFromFileLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) 
{
  const Dtype Zero = Dtype(0);
  CHECK_EQ(propagate_down.size(), bottom.size());

  for (int i = 0; i < propagate_down.size(); ++i) {
    if (propagate_down[i]) { 
      // NOT_IMPLEMENTED; 
      caffe_set(bottom[i]->count(), Zero, bottom[i]->mutable_cpu_diff());
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(LoadBboxFromFileLayer);
#endif

INSTANTIATE_CLASS(LoadBboxFromFileLayer);
REGISTER_LAYER_CLASS(LoadBboxFromFile);

}  // namespace caffe