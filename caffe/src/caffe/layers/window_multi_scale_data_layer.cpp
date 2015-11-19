// CopyRight ZhuJin Liang 2015

#include <map>
#include <deque>
#include <string>
#include <vector>
#include <fstream>  // NOLINT(readability/streams)
#include <utility>
#include <sstream>
#include <algorithm>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "caffe/layer.hpp"
#include "caffe/common.hpp"
#include "caffe/data_layers.hpp"
#include "caffe/global_variables.hpp"
#include "caffe/wanglan_face_shoulders_layers.hpp"

#include "caffe/util/rng.hpp"
#include "caffe/util/util_img.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/pose_tool.hpp"
#include "caffe/util/util_others.hpp"
#include "caffe/util/util_read_anno.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/util_pre_define.hpp"

using std::string;
using std::map;
using std::pair;
using std::deque;

namespace caffe {

template <typename Dtype>
WindowMultiScaleDataLayer<Dtype>::~WindowMultiScaleDataLayer<Dtype>() {
  this->JoinPrefetchThread();
}

template <typename Dtype>
void WindowMultiScaleDataLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
  BaseDataLayer<Dtype>::LayerSetUp(bottom, top);

  // Now, start the prefetch thread. Before calling prefetch, we make two
  // cpu_data calls so that the prefetch thread does not accidentally make
  // simultaneous cudaMalloc calls when the main thread is running. In some
  // GPUs this seems to cause failures if we do not so.
  LOG(INFO) << "Initialize data/images/labels/... mutable data...";
  this->prefetch_data_->mutable_cpu_data();
  this->prefetch_label_->mutable_cpu_data();
  if(this->with_bbox_) {
    this->prefetch_bbox_->mutable_cpu_data();
  }

  DLOG(INFO) << "Initializing prefetch";
  this->CreatePrefetchThread();
  DLOG(INFO) << "Prefetch initialized.";
}

/// Prepare for Initialization
template <typename Dtype>
void WindowMultiScaleDataLayer<Dtype>::DataLayerSetUp(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) 
{
  const WindowMultiScaleDataParameter window_multi_scale_data_param = 
      this->layer_param_.window_multi_scale_data_param();
  // bbox regression
  with_bbox_ = window_multi_scale_data_param.with_bbox();

  CHECK_EQ(bottom.size(), 0) 
      << "Window data Layer takes no input blobs.";
  CHECK_EQ(top.size(), with_bbox_ ? 3 : 2) 
      << "Window data Layer prodcues two blobs as output.";
  LOG(INFO);
  LOG(INFO) << "Window Multi-scale data layer, positive fraction: "
      << window_multi_scale_data_param.pos_fraction();
  LOG(INFO) << "Window Multi-scale data layer, hard sample fraction: "
      << window_multi_scale_data_param.hard_sample_fraction();
  LOG(INFO);
  // valid distance | invalid distance
  const int valid_distance = 
      window_multi_scale_data_param.valid_distance();
  const int invalid_distance = 
      window_multi_scale_data_param.has_invalid_distance() ?
      window_multi_scale_data_param.invalid_distance() : 
      window_multi_scale_data_param.valid_distance();
  CHECK_LE(valid_distance, invalid_distance) 
      << "Valid distance should be not greater than invalid distance";
  LOG(INFO) << "Valid distance is " << valid_distance;
  LOG(INFO) << "Invalid distance is " << invalid_distance;

  /// Read all the samples
  const int key_point_counts = 
      window_multi_scale_data_param.key_point_counts();
  {
    // See src/caffe/util/util_read_anno.cpp
    ReadAnnotations_hs(window_multi_scale_data_param.source().c_str(), 
        samples_, 
        samples_tags_,
        key_point_counts
    );
    LOG(INFO) << "Number of samples: " << samples_.size();
    LOG(INFO) << "Number of samples' tags: " << samples_tags_.size();
    LOG(INFO);

    // Get each target's diagonal length of bbox
    const bool has_standard_bbox_diagonal_len = 
        window_multi_scale_data_param.has_standard_bbox_diagonal_len();
    if(has_standard_bbox_diagonal_len) {
      const int standard_bbox_diagonal_len =
          window_multi_scale_data_param.standard_bbox_diagonal_len();
      // ??
      // See src/caffe/util/util_read_anno.cpp
      GetAllBBoxStandardScale(
          samples_, 
          key_point_counts,
          standard_bbox_diagonal_len, 
          bboxes_standard_scale_
      );
    }

    /// Visualize bbox
    bool visual_bbox = window_multi_scale_data_param.has_visual_bbox_path();
    if(visual_bbox) {
      const string img_ext = "jpg";
      const int vlen = key_point_counts * 2;
      // "/home/alan/workspace/heat-map/prototxt/cache/"
      const string visual_bbox_path = 
          window_multi_scale_data_param.visual_bbox_path();
      const string imgs_folder = 
          window_multi_scale_data_param.imgs_folder();

      for (int k = 0; k < samples_.size(); ++k) {
        cv::Mat img = cv::imread(imgs_folder + "/" + samples_[k].first + img_ext);
         vector<vector<float> > bboxes;
         // Get bounding boxes from coordinates
         // See src/caffe/util/util_read_anno.cpp
         GetBBoxes(samples_[k].second, key_point_counts, bboxes);
         // Check
         CHECK_EQ(bboxes.size(), samples_[k].second.size() / vlen);
         // start visualizing
         for (int j = 0; j < bboxes.size(); ++j) {
           // float w = bboxes[j][0] - bboxes[j][2];
           // float h = bboxes[j][1] - bboxes[j][3];
           // float len = std::sqrt(w * w + h * h);
           cv::rectangle(
              img, 
              cv::Point(bboxes[j][0], bboxes[j][1]), 
              cv::Point(bboxes[j][2], bboxes[j][3]),
              cv::Scalar(255, 6, 12));
         }
        cv::imwrite(visual_bbox_path + samples_[k].first + img_ext, img);
      }
    } 
  }

  /// Read hard sample
  {
    hard_samples_.clear();
    if(window_multi_scale_data_param.has_hard_sample_source()) {
      const string hard_sample_source = 
          window_multi_scale_data_param.hard_sample_source();
      // See src/caffe/util/util_read_anno.cpp
      ReadWindows(hard_sample_source.c_str(), hard_samples_);
      // print info
      LOG(INFO) << "hard_sample_source: " << hard_sample_source;
      LOG(INFO) << "Current number of hard samples: " << hard_samples_.size();
    } else {
      LOG(INFO) << "Here does not need hard sample source...";
    }
    LOG(INFO);
  }

  /// Read key points
  {
    const string key_points_file = 
        window_multi_scale_data_param.key_points_file();
    // See src/caffe/util/util_read_anno.cpp
    ReadKeyPoints(key_points_file.c_str(), key_point_idxs_, true);
    CHECK(key_point_idxs_.size() != 0)
      << "There should be at least one key point to be used.";
    LOG(INFO) << "key_points_file: " << key_points_file;
    //
    for(int kp = 0; kp < key_point_idxs_.size(); kp++) {
      LOG(INFO) << "kp: " << kp << ", id: " << key_point_idxs_[kp];
    }
    LOG(INFO);
  }

  /// Read standard length (must have one of them)
  {
    if(!window_multi_scale_data_param.has_standard_bbox_diagonal_len()) {
      const string standard_len_file = 
          window_multi_scale_data_param.standard_len_file();
      ReadStandardLengths(standard_len_file.c_str(), standard_len_, true);
    }

    CHECK(window_multi_scale_data_param.has_standard_bbox_diagonal_len()
        || standard_len_.size() != 0)
      << "There should be at least one standard length to be used "
      << "or has bbox diagonal length.";
  }

  /// Get rescale interval (continous or discrete)
  {
    is_range_scale_ = window_multi_scale_data_param.is_range_scale();
    if(is_range_scale_) {
      scale_lower_limit_ =
          window_multi_scale_data_param.scale_lower_limit();
      scale_upper_limit_ =
          window_multi_scale_data_param.scale_upper_limit();
      CHECK_GE(scale_upper_limit_, scale_lower_limit_) 
          << "Error scale range.";
      LOG(INFO) << "Scale range: [" << scale_lower_limit_ 
          << ", " << scale_upper_limit_ << "]";
    } else {
      const string scales_file = 
          window_multi_scale_data_param.scales_file();
      // See src/caffe/util/util_read_anno.cpp
      ReadScales(scales_file.c_str(), scales_candidate_, true);
      if(scales_candidate_.size() == 0) {
        scales_candidate_.push_back(1);
      }
    }
  }

  // Images|Labels|Bboxes Info
  const int batch_size = window_multi_scale_data_param.batch_size();
  const int height = window_multi_scale_data_param.patch_height();
  const int width = window_multi_scale_data_param.patch_width();
  
  top[0]->Reshape(batch_size, 3, height, width);
  prefetch_data_.reset(new Blob<Dtype>(batch_size, 3, height, width));
  LOG(INFO) << "Output data size";
  LOG(INFO) << "num: " << top[0]->num();
  LOG(INFO) << "channels: " << top[0]->channels();
  LOG(INFO) << "height: " << top[0]->height();
  LOG(INFO) << "width: " << top[0]->width();
  
  top[1]->Reshape(batch_size, key_point_idxs_.size(), 1, 1);
  prefetch_label_.reset(new Blob<Dtype>(batch_size, 
      key_point_idxs_.size(), 1, 1));
  LOG(INFO) << "Output labels' size";
  LOG(INFO) << "num: " << top[1]->num();
  LOG(INFO) << "channels: " << top[1]->channels();
  LOG(INFO) << "height: " << top[1]->height();
  LOG(INFO) << "width: " << top[1]->width();

  if(with_bbox_) {
    top[2]->Reshape(batch_size, 4, 1, 1);
    prefetch_bbox_.reset(new Blob<Dtype>(batch_size, 4, 1, 1));
    LOG(INFO) << "Output data size";
    LOG(INFO) << "num: " << top[2]->num();
    LOG(INFO) << "channels: " << top[2]->channels();
    LOG(INFO) << "height: " << top[2]->height();
    LOG(INFO) << "width: " << top[2]->width();
  }

  batch_reuse_count = window_multi_scale_data_param.batch_reuse_count();
  if(batch_reuse_count == 0) {
    batch_reuse_count = 1;
  }
  /// ??
  cur_batch_use_count = batch_reuse_count;
}

template <typename Dtype>
void WindowMultiScaleDataLayer<Dtype>::PerturbedCoordsBias() {
  /// TODO
}

template <typename Dtype>
void WindowMultiScaleDataLayer<Dtype>::ShuffleImages() {
  /// TODO
  // caffe::rng_t* prefetch_rng =
  //     static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  // shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

// This function is used to create a thread that prefetches the data.
// At each iteration, sample N windows where N*p are foreground (object)
// windows and N*(1-p) are background (non-object) windows
template <typename Dtype>
void WindowMultiScaleDataLayer<Dtype>::InternalThreadEntry() {
  const WindowMultiScaleDataParameter window_multi_scale_data_param 
      = this->layer_param_.window_multi_scale_data_param();
  const bool with_bbox = this->with_bbox_;
  Dtype* top_bbox = with_bbox ? this->prefetch_bbox_->mutable_cpu_data() : NULL;
  // Get top data|labels
  Blob<Dtype>* top_data_blob = this->prefetch_data_.get();
  Dtype* top_data = this->prefetch_data_->mutable_cpu_data();
  Dtype* top_label = this->prefetch_label_->mutable_cpu_data();

  const int batch_size = 
      window_multi_scale_data_param.batch_size();
  const float pos_fraction = 
      window_multi_scale_data_param.pos_fraction();
  const float hard_sample_fraction = 
      window_multi_scale_data_param.hard_sample_fraction();
  const int key_point_count = 
      window_multi_scale_data_param.key_point_counts();
  // 
  const vector<pair<string, vector<float> > >& samples = this->samples_;
  // const vector<pair<string, vector<int> > >& samples_tags = this->samples_tags_;

  const vector<int>& key_point_idxs = this->key_point_idxs_;
  const vector<vector<int> >& standard_len = this->standard_len_;
  const deque<pair<string, vector<float> > >& 
      hard_samples = this->hard_samples_;
  const vector<vector<float> >& bboxes_standard_scale = 
      this->bboxes_standard_scale_;

  const bool is_range_scale = this->is_range_scale_;
  const float scale_lower_limit = this->scale_lower_limit_;
  const float scale_upper_limit = this->scale_upper_limit_;
  const vector<float>& scales = this->scales_candidate_;

  const caffe::rng_t* const prefetch_rng = 
      static_cast<caffe::rng_t*>(this->prefetch_rng_->generator());

  const int patch_height = this->prefetch_data_->height();
  const int patch_width = this->prefetch_data_->width();
  // const int patch_channels = this->prefetch_data_->channels();
  const int half_patch_height =  patch_height / 2;
  const int half_patch_width = patch_width/ 2;
  const cv::Size standard_crop_size(patch_width, patch_height);

  const int standard_valid_distance = 
      window_multi_scale_data_param.valid_distance();
  const bool has_invalid_distance = 
      window_multi_scale_data_param.has_invalid_distance();
  const int standard_invalid_distance =  has_invalid_distance ?
      window_multi_scale_data_param.invalid_distance() : 
      window_multi_scale_data_param.valid_distance();
  CHECK_LE(standard_valid_distance, standard_invalid_distance) 
      << "Valid distance should be not greater than invalid distance";

  const bool is_fill = window_multi_scale_data_param.is_fill();
  const int fill_value = window_multi_scale_data_param.fill_value();

  // zero out batch
  memset(top_data, 0, sizeof(Dtype) * this->prefetch_data_->count());
  memset(top_label, 0, sizeof(Dtype) * this->prefetch_label_->count());
  if(with_bbox) {
    for (int i = 0; i < this->prefetch_bbox_->count(); ++i) {
      top_bbox[i] = -1;
    }
  }

  const int num_pos = static_cast<int>(
      static_cast<float>(batch_size) * pos_fraction);

  /// Visualize
  char output_path[PATH_MAX];
  const string img_ext = ".jpg";
  bool visual_bbox = 
      window_multi_scale_data_param.has_visual_bbox_path();
  const bool pic_print = ((getenv("PIC_PRINT") != NULL) 
      && (getenv("PIC_PRINT")[0] == '1'));
  if(visual_bbox && pic_print) {
    const string visual_bbox_path = 
        window_multi_scale_data_param.visual_bbox_path();
    
    const string cache_path1 = "cache/WindowMultiScaleDataLayer/pic";
    const string cache_dir1 = visual_bbox_path + cache_path1;
    sprintf(output_path, "%s", cache_dir1.c_str());
    CreateDir(output_path);
    LOG(INFO) << "output_path: " << output_path;
  
    const string cache_path2 = "cache/WindowMultiScaleDataLayer";
    const string cache_dir2 = visual_bbox_path + cache_path2;
    sprintf(output_path, "%s", cache_dir2.c_str());
  }

  char path[PATH_MAX];
  const string img_foler = window_multi_scale_data_param.imgs_folder();

  // ##################################################################
  /// Generate positive sample
  int item_id = 0;
  vector<float> crop_coords(4, 0);
  float center_x, center_y;
  const int pKpc = key_point_count * 2;
  // int x1 = 0, y1 = 0, x2 = 0, y2 = 0;
  /// Positive
  while (item_id < num_pos) {
    const int sample_idx = this->PrefetchRand() % samples.size();
    const pair<string, vector<float> >& sample = samples[sample_idx];

    if(sample.second.size() == 0) {
      LOG(ERROR) << "Sample " << sample.first << "has no annotations";
      continue;
    }

    // there may be multi-target in one image, randomly select one target
    const vector<float>& coords = sample.second;
    const int clen = coords.size();
    // randomly select one person from an image.
    const int objidx = this->PrefetchRand() % (clen / pKpc);
    // randomly select one key point
    const int p_idx = this->PrefetchRand() % key_point_idxs.size();
    const int key_point_idx = key_point_idxs[p_idx];
    const int idx = key_point_idx + objidx * key_point_count;

    center_x = coords[idx * 2];
    center_y = coords[idx * 2 + 1];
    if(center_x < 0 || center_y < 0) {
      continue;
    }

    sprintf(path, "%s/%s%s", img_foler.c_str(), 
        sample.first.c_str(), img_ext.c_str());
    cv::Mat cv_img = cv::imread(path, CV_LOAD_IMAGE_COLOR);
    if(!cv_img.data) {
      LOG(ERROR)<< "Could not open or find file " << path;
      return;
    }
    if(cv_img.empty()) {
      LOG(ERROR) << "Empty image: " << path;
      continue;
    }

    if(center_x >= cv_img.cols || center_y >= cv_img.rows) continue;

    /// Get the scale that the input image to standard image
    float scale_to_standard = 1;
    if(window_multi_scale_data_param.has_standard_bbox_diagonal_len()) {
      scale_to_standard = bboxes_standard_scale[sample_idx][objidx];
    } else {
      for (int standard_len_i = 0; standard_len_i < standard_len.size(); ++standard_len_i) {
        int pt1 = standard_len[standard_len_i][0] + objidx * key_point_count;
        int pt2 = standard_len[standard_len_i][1] + objidx * key_point_count;
        if(coords[pt1 * 2] != -1 && coords[pt2 * 2] != -1) {
          const int len = standard_len[standard_len_i][2];
          float x_diff = fabs(coords[pt2 * 2] - coords[pt1 * 2]);
          float y_diff = fabs(coords[pt2 * 2 + 1] - coords[pt1 * 2 + 1]);
          float pt_diff = x_diff * x_diff + y_diff * y_diff;
          //
          if(pt_diff < 25) {
            LOG(INFO)<< sample.first << " " << pt1 << " " << pt2
            << " " << coords[pt2 * 2] << " " << coords[pt1 * 2]
            << " " << coords[pt2 * 2 + 1] << " " << coords[pt1 * 2 + 1];
            continue;
          }

          pt_diff = sqrtf(pt_diff);
          scale_to_standard = len / pt_diff;

          break;
        }
      }
    }

    /// Add perturbation
    if(standard_valid_distance > 0) {
      int v_dist = standard_valid_distance;
      v_dist = v_dist * v_dist;
      for (int random_counts = 0; random_counts < 10; ++random_counts) {
        int delta_x = this->PrefetchRand() % standard_valid_distance;
        int delta_y = this->PrefetchRand() % standard_valid_distance;
        int dist = delta_x * delta_x + delta_y * delta_y;
        if(dist <= v_dist) {
          if(this->PrefetchRand() % 2 == 0) delta_x = -delta_x;
          if(this->PrefetchRand() % 2 == 0) delta_y = -delta_y;
          //
          float new_center_x = center_x + (delta_x / scale_to_standard);
          if(new_center_x >= 0 && new_center_x < cv_img.cols) {
            center_x = new_center_x;
          }
          //
          float new_center_y = center_y + (delta_y / scale_to_standard);
          if(new_center_y >= 0 && new_center_y < cv_img.rows) {
            center_y = new_center_y;
          }

          break;
        }
      }
    }

    /// the original patch size is (patch_width, patch_height)
    /// now add the random_scale on it
    /// the result patch size is (patch_width * random_scale, patch_height * random_scale)
    float random_scale;
    if(is_range_scale) {
      random_scale = this->PrefetchRand() / 
          static_cast<float>(prefetch_rng->max());
      random_scale *= (scale_upper_limit - scale_lower_limit);
      random_scale += scale_lower_limit;
    } else {
      random_scale = scales[this->PrefetchRand() % scales.size()];
    }

    const cv::Size crop_size(
        random_scale * standard_crop_size.width / scale_to_standard,
        random_scale * standard_crop_size.height / scale_to_standard
    );
    crop_coords[0] = center_x - crop_size.width / 2;
    crop_coords[1] = center_y - crop_size.height / 2;
    crop_coords[2] = center_x + crop_size.width / 2;
    crop_coords[3] = center_y + crop_size.height / 2;

    cv::Mat cv_input_patch;
    //截取图片，然后复制到top_data中
    CropAndResizePatch(cv_img, 
        cv_input_patch, 
        crop_coords, 
        standard_crop_size, 
        is_fill, 
        fill_value
    );
    ImageDataToBlob(
        top_data_blob, 
        item_id, 
        cv_input_patch
    ); //

    /// Set the label (heat map)
    /// one key point store in one channel.
    /// if there is at least one key point falling in the circle
    /// which center is (center_x, center_y), radius is valid_distance,
    /// set the correspond channel value 1, else 0
    const float valid_distance = standard_valid_distance / scale_to_standard;
    for (int label_i = 0; label_i < key_point_idxs.size(); ++label_i) {
      int top_index = item_id * key_point_idxs.size() + label_i;
      CHECK_EQ(top_label[top_index], 0)
          << "Duplicate Label Initializing...";
      CHECK_LT(top_index, this->prefetch_label_->count());

      int coords_i = key_point_idxs[label_i] * 2;
      for (; coords_i < coords.size(); coords_i += (key_point_count * 2)) {
        float pt_x = coords[coords_i];
        float pt_y = coords[coords_i + 1];
        // Check
        if(pt_x < 0 || pt_x >= cv_img.cols || pt_y < 0 
            || pt_y >= cv_img.rows) {
          continue;
        }
        // distance
        float diff = (pt_x - center_x) * (pt_x - center_x) 
            + (pt_y - center_y) * (pt_y - center_y);
        if(diff > valid_distance * valid_distance) continue;

        top_label[top_index] = 1;
        break;
      }
    }

  	// Set bbox
    if(with_bbox) {
  		vector<vector<float> > bboxes;
  		GetBBoxes(coords, key_point_count, bboxes);
  		// first subtract the top left corner of crop patch
  		top_bbox[item_id * 4 + 0] =  bboxes[objidx][0] - center_x;
  		top_bbox[item_id * 4 + 1] =  bboxes[objidx][1] - center_y;
  		top_bbox[item_id * 4 + 2] =  bboxes[objidx][2] - center_x;
  		top_bbox[item_id * 4 + 3] =  bboxes[objidx][3] - center_y;
  		// and the rescale according the crop patch
  		float bbox_scale_w = standard_crop_size.width 
          / static_cast<float>(crop_size.width);
  		float bbox_scale_h = standard_crop_size.height 
          / static_cast<float>(crop_size.height);
  		top_bbox[item_id * 4 + 0] *= bbox_scale_w;
  		top_bbox[item_id * 4 + 1] *= bbox_scale_h;
  		top_bbox[item_id * 4 + 2] *= bbox_scale_w;
  		top_bbox[item_id * 4 + 3] *= bbox_scale_h;
    }

    if(pic_print) {
      LOG(INFO) << "Saving pos sample: " << sample.first
          << ", person index: " << objidx
          << ", key point: " << key_point_idx
          << ", scale: " << random_scale;
      static int pos_idx = 0;
      sprintf(path, "%s/pic/pos_%d_%s_p%02d_s%f_input%s", 
          output_path, 
          pos_idx,
          sample.first.c_str(), 
          key_point_idx, 
          random_scale,
          img_ext.c_str());
      imwrite(path, cv_input_patch);

      sprintf(path, "%s/pic/pos_%d_%s_p%02d_s%f_original%s",
          output_path, 
          pos_idx,
          sample.first.c_str(), 
          key_point_idx, 
          random_scale,
          img_ext.c_str());
      // Get valid coordinates
      int x1 = MIN(cv_img.cols - 1, MAX(0, crop_coords[0]));
      int x2 = MIN(cv_img.cols - 1, MAX(0, crop_coords[2]));
      int y1 = MIN(cv_img.rows - 1, MAX(0, crop_coords[1]));
      int y2 = MIN(cv_img.rows - 1, MAX(0, crop_coords[3]));
      // Draw
      cv::Rect roi(x1, y1, x2 - x1 + 1, y2 - y1 + 1);
      cv::rectangle(cv_img, roi, cv::Scalar(0, 0, 255));
      cv::circle(cv_img, 
          cv::Point(coords[idx * 2], coords[idx * 2 + 1]), 
          3, cv::Scalar(255, 0, 0), 2, 8);
      cv::circle(cv_img, 
          cv::Point(center_x, center_y), 3, 
          cv::Scalar(0, 255, 0), 2, 8);
      // Visualize
      if(window_multi_scale_data_param.has_standard_bbox_diagonal_len()) {
        vector<vector<float> > bboxes;
        GetBBoxes(sample.second, key_point_count, bboxes);
        cv::rectangle(cv_img, 
            cv::Point(bboxes[objidx][0], bboxes[objidx][1]),
            cv::Point(bboxes[objidx][2], bboxes[objidx][3]), 
            cv::Scalar(255, 12, 16));
      }
      imwrite(path, cv_img);

      sprintf(path, "%s/pic/pos_%d_%s_p%02d_s%f.txt", 
          output_path, 
          pos_idx,
          sample.first.c_str(), 
          key_point_idx, 
          random_scale);
      std::ofstream out_file(path);
      out_file << "image size: " << cv_img.cols 
          << ", " << cv_img.rows << std::endl;
      out_file << "bbox: " << x1 << ", " << y1 
          << ", " << x2 << ", " << y2 << std::endl;
      out_file << "labels:";
      //
      for (int label_i = 0; label_i < key_point_idxs.size(); ++label_i) {
        int idx = item_id * key_point_idxs.size() + label_i;
        out_file << " " << top_label[idx];
      }
      out_file << std::endl;
      out_file.close();

      ++pos_idx;
    }

    ++item_id;
  }

  // #############################################################################
  /// Hard negative
  int num_hard_sample = MIN(hard_samples.size(),
      static_cast<int>((batch_size - num_pos) * hard_sample_fraction));
  //
  while (item_id < batch_size && num_hard_sample != 0) {
    const pair<string, vector<float> >& sample = 
        hard_samples[this->PrefetchRand() % hard_samples.size()];

    sprintf(path, "%s/%s%s", 
        img_foler.c_str(), 
        sample.first.c_str(), 
        img_ext.c_str());
    cv::Mat cv_img = cv::imread(path, CV_LOAD_IMAGE_COLOR);
    if(!cv_img.data) {
      LOG(ERROR)<< "Could not open or find file " << path;
      return;
    }
    if(cv_img.empty()) {
      LOG(ERROR) << "Empty image: " << path;
      continue;
    }

    cv::Mat cv_input_patch;
    CropAndResizePatch(
        cv_img, 
        cv_input_patch, 
        sample.second, 
        standard_crop_size, 
        is_fill, 
        fill_value
    );
    ImageDataToBlob(top_data_blob, item_id, cv_input_patch);

    /// Visualize
    bool visual_bbox = 
        window_multi_scale_data_param.has_visual_bbox_path();
    if(visual_bbox && pic_print) {
      LOG(INFO)<< "Saving hard sample: " << sample.first << "...";
      static int hard_idx = 0;
      sprintf(path, "%s/pic/hard_%d_%s_input%s", 
          output_path, 
          hard_idx, 
          sample.first.c_str(),
          img_ext.c_str());
      imwrite(path, cv_input_patch);

      sprintf(path, "%s/pic/hard_%d_%s_original%s", 
          output_path, 
          hard_idx, 
          sample.first.c_str(),
          img_ext.c_str());
      // 
      int x1 = MIN(cv_img.cols - 1, MAX(0, sample.second[0]));
      int y1 = MIN(cv_img.rows - 1, MAX(0, sample.second[1]));
      int x2 = MIN(cv_img.cols - 1, MAX(0, sample.second[2]));
      int y2 = MIN(cv_img.rows - 1, MAX(0, sample.second[3]));
      // Draw
      cv::Rect roi(x1, y1, x2 - x1 + 1, y2 - y1 + 1);
      cv::rectangle(cv_img, roi, cv::Scalar(0, 0, 255), 5);
      cv::circle(cv_img, cv::Point((sample.second[0] + sample.second[2] / 2), 
          (sample.second[1] + sample.second[3] / 2)),
          3, cv::Scalar(255, 0, 0), 2, 8);
      imwrite(path, cv_img);
      // 
      sprintf(path, "%s/pic/hard_%d_%s.txt", 
          output_path, 
          hard_idx, 
          sample.first.c_str());
      std::ofstream out_file(path);
      out_file << "image size: " << cv_img.cols 
          << ", " << cv_img.rows << std::endl;
      out_file << "bbox: " << x1 << ", " << y1 
          << ", " << x2 << ", " << y2 << std::endl;
      out_file << "labels:";
      for (int label_i = 0; label_i < key_point_idxs.size(); ++label_i) {
        int idx = item_id * key_point_idxs.size() + label_i;
        out_file << " " << top_label[idx];
      }
      out_file << std::endl;
      out_file.close();

      ++hard_idx;
    }

    ++item_id;
    --num_hard_sample;
  }

  // #############################################################################
  /// Negative
  while (item_id < batch_size) {
    const int sample_idx = this->PrefetchRand() % samples.size();
    const pair<string, vector<float> >& sample = samples[sample_idx];
    vector<float> coords(sample.second);

    sprintf(path, "%s/%s%s", 
        img_foler.c_str(), 
        sample.first.c_str(),
        img_ext.c_str());
    const cv::Mat cv_img = cv::imread(path, CV_LOAD_IMAGE_COLOR);
    if(!cv_img.data) {
      LOG(ERROR)<< "Could not open or find file " << path;
      return;
    }
    if(cv_img.empty()) {
      LOG(ERROR) << "Empty image: " << path;
      continue;
    }

    int range_x = cv_img.cols - standard_crop_size.width + 1;
    int range_y = cv_img.rows - standard_crop_size.height + 1;
    if(range_x <= 1 || range_y <= 1) {
      continue;
    }

    // try to generate the negative window
    // if the generated window contain key point, and generated again.
    // at almost ten time for one image.
    // int x1 = -1, x2 = -1, y1 = -1, y2 = -1;
    for (int i = 0; i < 10; ++i) {
      int center_x = this->PrefetchRand() % range_x + half_patch_width;
      int center_y = this->PrefetchRand() % range_y + half_patch_height;

      // the distance between the center of selected patch and each key point
      // should greater than invalid_distance,
      // besides, in order to crop the patch, we need to scale the images
      // the scale we use is the scale of nearest target.
      bool ok = true;
      int nearest_id = -1;
      float nearest_dis = -1e6;
      for (int key_point_idx = 0; key_point_idx < key_point_idxs.size() && ok; 
          ++key_point_idx) 
      {
        for (int target_i = 0; target_i < bboxes_standard_scale[sample_idx].size(); 
            ++target_i) 
        {
          int coords_i = key_point_idx + target_i * key_point_count;
          coords_i *= 2;
          float pt_x = coords[coords_i];
          float pt_y = coords[coords_i + 1];
          // Check
          if(pt_x < 0 || pt_x >= cv_img.cols || pt_y < 0 || pt_y >= cv_img.rows) {
            continue;
          }

          // get the distance in standard scale
          float diff = (pt_x - center_x) * (pt_x - center_x) 
              + (pt_y - center_y) * (pt_y - center_y);
          diff = std::sqrt(diff) * bboxes_standard_scale[sample_idx][target_i];

          if(diff <= standard_invalid_distance) {
            ok = false;
            break;
          }

          if(nearest_id == -1 || nearest_dis > diff) { 
            nearest_id = target_i;
            nearest_dis = diff;
          }
        }
      }
      // Not found
      if(!ok) {
        continue;
      }

      if(nearest_id == -1 && bboxes_standard_scale[sample_idx].size() != 0) {
        nearest_id = this->PrefetchRand() % bboxes_standard_scale[sample_idx].size();
      }

      // Get the scale of its nearest target
      float scale_to_standard = 1;
      if(window_multi_scale_data_param.has_standard_bbox_diagonal_len()) {
        if(nearest_id != -1) {
          scale_to_standard = bboxes_standard_scale[sample_idx][nearest_id];
        } else {
          scale_to_standard = 1;
        }
      } else {
        for (int standard_len_i = 0; standard_len_i < standard_len.size(); 
            ++standard_len_i) 
        {
          int pt1 = standard_len[standard_len_i][0] 
              + nearest_id * key_point_count;
          int pt2 = standard_len[standard_len_i][1] 
              + nearest_id * key_point_count;
          if(coords[pt1 * 2] != -1 && coords[pt2 * 2] != -1) {
            const int len = standard_len[standard_len_i][2];
            float x_diff = fabs(coords[pt2 * 2] - coords[pt1 * 2]);
            float y_diff = fabs(coords[pt2 * 2 + 1] - coords[pt1 * 2 + 1]);
            float pt_diff = x_diff * x_diff + y_diff * y_diff;
            if(pt_diff < 25) {
              LOG(INFO)<< sample.first << " " << pt1 << " " << pt2
                  << " " << coords[pt2 * 2] << " " << coords[pt1 * 2]
                  << " " << coords[pt2 * 2 + 1] << " " << coords[pt1 * 2 + 1];
              continue;
            }

            pt_diff = sqrtf(pt_diff);
            scale_to_standard = len / pt_diff;
            break;
          }
        }
      }

      crop_coords[0] = center_x - half_patch_width / scale_to_standard;
      crop_coords[1] = center_y - half_patch_height / scale_to_standard;
      crop_coords[2] = center_x + half_patch_width / scale_to_standard;
      crop_coords[3] = center_y + half_patch_height / scale_to_standard;

      cv::Mat cv_input_patch;
      CropAndResizePatch(cv_img, 
          cv_input_patch, 
          crop_coords, 
          standard_crop_size, 
          is_fill, 
          fill_value);
      // 这里可能对我的问题不一定适合，
      // 比如截取的图片在四个角周围的时候，
      // 那么就只能填充个四分之一，就跟肩膀有点像了，
      ImageDataToBlob(top_data_blob, item_id, cv_input_patch);

      if(visual_bbox && pic_print) {
        LOG(INFO)<< "Saving neg sample: " << sample.first << "...";

        static int neg_idx = 0;
        sprintf(path, "%s/pic/neg_%d_%s_input%s", 
            output_path, 
            neg_idx, 
            sample.first.c_str(), 
            img_ext.c_str());
        imwrite(path, cv_input_patch);

        cv::Mat cv_img_print = cv_img.clone();
        sprintf(path, "%s/pic/neg_%d_%s_original%s", 
            output_path, 
            neg_idx, 
            sample.first.c_str(),
            img_ext.c_str());
        // 
        int x1 = MIN(cv_img.cols - 1, MAX(0, 
            center_x - half_patch_width / scale_to_standard));
        int x2 = MIN(cv_img.cols - 1, MAX(0, 
            center_x + half_patch_width / scale_to_standard));
        int y1 = MIN(cv_img.rows - 1, MAX(0, 
            center_y - half_patch_height / scale_to_standard));
        int y2 = MIN(cv_img.rows - 1, MAX(0, 
            center_y + half_patch_height / scale_to_standard));
        // Draw
        cv::Rect roi(x1, y1, x2 - x1 + 1, y2 - y1 + 1);
        cv::rectangle(cv_img_print, roi, cv::Scalar(0, 0, 255), 5);
        cv::circle(cv_img_print, cv::Point(center_x, center_y), 3, 
            cv::Scalar(255, 0, 0), 2, 8);
        if(window_multi_scale_data_param.has_standard_bbox_diagonal_len()) {
          vector<vector<float> > bboxes;
          GetBBoxes(sample.second, key_point_count, bboxes);
          for (int i = 0; i < bboxes.size(); ++i) {
            cv::rectangle(cv_img_print, cv::Point(bboxes[i][0], bboxes[i][1]),
                cv::Point(bboxes[i][2], bboxes[i][3]), cv::Scalar(255, 0, 0));
          }
        }
        imwrite(path, cv_img_print);

        sprintf(path, "%s/pic/neg_%d_%s.txt", 
            output_path, 
            neg_idx, 
            sample.first.c_str());
        std::ofstream out_file(path);
        out_file << "image size: " << cv_img_print.cols 
            << ", " << cv_img_print.rows << std::endl;
        out_file << "bbox: " << x1 << ", " << y1 
            << ", " << x2 << ", " << y2 << std::endl;
        out_file << "labels:";
        for (int label_i = 0; label_i < key_point_idxs.size(); ++label_i) {
          int idx = item_id * key_point_idxs.size() + label_i;
          out_file << " " << top_label[idx];
        }
        out_file << std::endl;
        out_file.close();

        ++neg_idx;
      }

      ++item_id;
      /// ??
      // break;
    }
  }
  return;
}

template <typename Dtype>
unsigned int WindowMultiScaleDataLayer<Dtype>::PrefetchRand() {
  CHECK(prefetch_rng_);
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  return (*prefetch_rng)();
}

template <typename Dtype>
void WindowMultiScaleDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) 
{
  if(cur_batch_use_count < batch_reuse_count) {
    ++cur_batch_use_count;
    return;
  }
  cur_batch_use_count = 1;

  /// First, join the thread
  this->JoinPrefetchThread();
  DLOG(INFO) << "Thread joined";

  // // Used for Accuracy and Visualization layers
  // // place in `Forward_cpu` to keep synchronization with top blobs
  // // when use `imgidxs` and `images_paths` in other layers
  // GlobalVars::set_objidxs(this->objidxs_);
  // GlobalVars::set_imgidxs(this->imgidxs_);
  // GlobalVars::set_images_paths(this->images_paths_);

  /// Copy the data
  // image
  // Reshape to loaded data.
  top[0]->Reshape(
      this->prefetch_data_->num(), 
      this->prefetch_data_->channels(),
      this->prefetch_data_->height(), 
      this->prefetch_data_->width()
  );
  /// Copy the data
  caffe_copy(
      this->prefetch_data_->count(), 
      this->prefetch_data_->cpu_data(), 
      top[0]->mutable_cpu_data()
  );
  /// Copy the labels
  caffe_copy(
      this->prefetch_label_->count(), 
      this->prefetch_label_->cpu_data(), 
      top[1]->mutable_cpu_data()
  );
  DLOG(INFO) << "Prefetch copied";
  /// Copy the bboxes
  if(with_bbox_) {
    caffe_copy(
        this->prefetch_bbox_->count(), 
        this->prefetch_bbox_->cpu_data(), 
        top[2]->mutable_cpu_data()
    );
  }
  /// Start a new prefetch thread
  DLOG(INFO) << "CreatePrefetchThread";
  this->CreatePrefetchThread();
}

INSTANTIATE_CLASS(WindowMultiScaleDataLayer);
REGISTER_LAYER_CLASS(WindowMultiScaleData);

}  // namespace caffe