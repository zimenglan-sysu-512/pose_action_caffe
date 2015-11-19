// Copyright 2014 BVLC and contributors.

#ifndef CAFFE_UTIL_READ_ANNO_H_
#define CAFFE_UTIL_READ_ANNO_H_

#include <string>
#include <vector>
#include <deque>
#include <utility>

#include "hdf5.h"
#include "hdf5_hl.h"
#include "caffe/proto/caffe.pb.h"
#include "google/protobuf/message.h"

#include "caffe/blob.hpp"

using std::string;
using std::vector;
using std::pair;
using std::deque;
using ::google::protobuf::Message;

namespace caffe {

void ReadAnnotations(
    const char* file, 
    vector<pair<string, vector<float> > >& samples,
    const int key_point_count = 72, 
    const bool warnEmpty = false);

inline void ReadAnnotations(
    const string& file, 
    vector<pair<string, vector<float> > >& samples,
    const int key_point_count = 72, 
    const bool warnEmpty = false) 
{
	ReadAnnotations(file.c_str(), samples, key_point_count, warnEmpty);
}

//Read head-shoulder key points
void ReadAnnotations_hs(
    const char* file, 
    vector<pair<string, vector<float> > >& samples,
		vector<pair<string, vector<int> > >& samples_tags,
		const int key_point_count = 3, 
    const bool warnEmpty= false);

// Mark
inline void ReadAnnotations_hs(
    const string& file, 
    vector<pair<string, vector<float> > >& samples,
		vector<pair<string, vector<int> > >& samples_tags,
		const int key_point_count = 3, 
    const bool warnEmpty = false) 
{
	ReadAnnotations_hs(file.c_str(), 
      samples,samples_tags, key_point_count, warnEmpty);
}


void ReadWindows(const char* file, deque<pair<string, 
    vector<float> > >& samples);

inline void ReadWindows(const string& file, 
    deque<pair<string, vector<float> > >& samples) 
{
	ReadWindows(file.c_str(), samples);
}

// Set flag true to output standard length info
void ReadStandardLengths(const char* file, 
    vector<vector<int> >& standard_len, const bool flag = false);

inline void ReadStandardLengths(const string& file, 
    vector<vector<int> >& standard_len, const bool flag = false) 
{
	ReadStandardLengths(file.c_str(), standard_len, flag);
}

// Set flag true to output key point info
void ReadKeyPoints(const char* file, 
    vector<int>& key_point_idxs, const bool flag = false);

inline void ReadKeyPoints(const string& file, 
    vector<int>& key_point_idxs, const bool flag = false) 
{
	ReadKeyPoints(file.c_str(), key_point_idxs, flag);
}

// Set flag true to output scale info
void ReadScales(const char* file, 
    vector<float>& scales, const bool flag = false);

inline void ReadScales(const string& file, 
    vector<float>& scales, const bool flag = false) 
{
	ReadScales(file.c_str(), scales, flag);
}

}  // namespace caffe

#endif   // CAFFE_UTIL_READ_ANNO_H_
