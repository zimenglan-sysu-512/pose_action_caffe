// Copyright 2014 BVLC and contributors.

#include <stdint.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/io/coded_stream.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

#include <algorithm>
#include <string>
#include <vector>
#include <fstream>  // NOLINT(readability/streams)

#include "caffe/common.hpp"
#include "caffe/util/util_read_anno.hpp"
#include "caffe/proto/caffe.pb.h"

using std::fstream;
using std::ios;
using std::max;
using std::string;

namespace caffe {

/// Read the image file name and key points
void ReadAnnotations(
    const char* file, 
    vector<pair<string, vector<float> > >& samples,
    const int key_point_count, 
    const bool warnEmpty) 
{
  std::ifstream input_file(file);
  CHECK(input_file.good()) 
      << "Failed to open annotation file " 
      << file << std::endl;

  samples.clear();
  string filename;
  string line;
  while(getline(input_file, line)) {
    if (line.empty()) continue;

    std::istringstream iss(line);
    iss >> filename;

    vector<float> coords;
    float t_coord;
    while(iss >> t_coord) {
      coords.push_back(t_coord);
    }
    /// Check
    if (coords.size() == 0 && warnEmpty) {
    	LOG(INFO) << "No annotations is set for image: " << filename;
    }
    CHECK(key_point_count != 0 && coords.size() % (key_point_count * 2) == 0)
      << "The number of key points is wrong: " << line;

    samples.push_back(make_pair(filename, coords));

  }
  // Close
  input_file.close();
}

/// Read the image file name and key points
void ReadAnnotations_hs(
    const char* file, 
    vector<pair<string, vector<float> > >& samples,
		vector<pair<string, vector<int> > >& samples_tags,
		const int key_point_count, 
    const bool warnEmpty) 
{
  std::ifstream input_file(file);
  CHECK(input_file.good()) 
      << "Failed to open annotation file " 
      << file << std::endl;

  samples.clear();
  samples_tags.clear();
  string filename;
  string line;
  while(getline(input_file, line)) {
    if (line.empty()) continue;

    std::istringstream iss(line);
    iss >> filename;

    int tag;
    float tx,ty;
    vector<int> tags;
    vector<float> coords;
    while(iss >> tx >> ty >> tag) {
      coords.push_back(tx);
      coords.push_back(ty);
      tags.push_back(tag);
    }
    /// Check
    if (coords.size() == 0 && tags.size() == 0 && warnEmpty) {
    	LOG(INFO) << "No annotations is set for image: " << filename;
    }
    CHECK(key_point_count != 0 
        && coords.size() % (key_point_count * 2) == 0)
        << "The number of key points is wrong: " << line;
    CHECK(tags.size() * 2 == coords.size()) 
        << "The number of key points is wrong: " << line;

    samples.push_back(make_pair(filename, coords));
    samples_tags.push_back(make_pair(filename, tags));
  }
  // Close 
  input_file.close();
}

// Read image file name and the bounding boxes
void ReadWindows(const char* file, 
    deque<pair<string, vector<float> > >& samples) 
{
  // Read all the samples
  std::ifstream input_file(file);
  CHECK(input_file.good()) 
      << "Failed to open window file " 
      << file << std::endl;

  string line;
  string filename;
  samples.clear();

  while(getline(input_file, line)) {
    if (line.empty()) continue;

    std::istringstream iss(line);
    iss >> filename;

    float t_coord;
    vector<float> coords;
    while(iss >> t_coord) {
      coords.push_back(t_coord);
    }

    CHECK(coords.size() == 4) 
      << "Each sample should has exactly 4dims bbox(x1, y1, x2, y2): " 
      << filename;
    samples.push_back(make_pair(filename, coords));
  }
  input_file.close();
}

/// Read standard length
void ReadStandardLengths(const char* file, 
    vector<vector<int> >& standard_len, const bool flag) 
{
  std::ifstream input_file(file);
  if (!input_file) {
    LOG(ERROR) << "Failed to open standard length file " 
        << file << std::endl;
  }

  standard_len.clear();
  vector<int> tmp(3, 0);
  while(input_file >> tmp[0] >> tmp[1] >> tmp[2]) {
    standard_len.push_back(tmp);
  }
  input_file.close();

  // print info
  if (!flag) return;
  std::ostringstream oss;
  for (int i = 0; i < standard_len.size(); ++i) {
    oss << standard_len[i][0] << " to " 
        << standard_len[i][1] << " is "
        << standard_len[i][2] << "; ";
  }
  LOG(INFO) << "There are " << standard_len.size()
      << " standard length: ";
  LOG(INFO) << oss.str();
  LOG(INFO);
}

/// Read key points
void ReadKeyPoints(const char* file, 
    vector<int>& key_point_idxs, const bool flag) 
{
  std::ifstream input_file(file);
  CHECK(input_file.good()) 
    << "Failed to open key point file " << file << std::endl;

  key_point_idxs.clear();
  int key_point;
  while(input_file >> key_point) {
    key_point_idxs.push_back(key_point);
  }
  input_file.close();

  CHECK(key_point_idxs.size() != 0) 
      << "There should be at least one key point to be used.";

  // print info
  if (!flag) return;
  std::ostringstream oss;
  for (int i = 0; i < key_point_idxs.size(); ++i) {
    oss << key_point_idxs[i] << " ";
  }
  LOG(INFO) << "There are " << key_point_idxs.size() 
      << " key points: ";
  LOG(INFO) << oss.str();
  LOG(INFO);
}

void ReadScales(const char* file, 
    vector<float>& scales, const bool flag) 
{
  std::ifstream input_file(file);
  CHECK(input_file.good()) 
      << "Failed to open key point file " << file << std::endl;

  scales.clear();
  float scale;
  while(input_file >> scale) {
    scales.push_back(scale);
  }
  input_file.close();

  CHECK(scales.size() != 0) 
      << "There should be at least one key point to be used.";

  // print info
  if (!flag) return;
  std::ostringstream oss;
  for (int i = 0; i < scales.size(); ++i) {
    oss << scales[i] << " ";
  }
  LOG(INFO) << "There are " << scales.size() << " scales: ";
  LOG(INFO) << oss.str();
  LOG(INFO);
}

}
  // namespace caffe
