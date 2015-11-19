// Copyright 2015 Zhu.Jin Liang


#ifndef CAFFE_UTIL_UTIL_COORDS_H_
#define CAFFE_UTIL_UTIL_COORDS_H_

#include <map>

#include "caffe/net.hpp"
#include "caffe/util/coords.hpp"

namespace caffe {

template <typename Dtype> class Net;

/*
 * @brief bottom的大小至少为0，找出在net中，
 * 				从bottom[0]映射到bottom[1]的映射关系
 */
template <typename Dtype>
DiagonalAffineMap<Dtype> GetMapBetweenFeatureMap(const vector<Blob<Dtype>*>& bottom,
		const Net<Dtype>* const net);

}  // namespace caffe

#endif  // CAFFE_UTIL_UTIL_COORDS_H_
