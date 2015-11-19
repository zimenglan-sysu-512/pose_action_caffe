// Copyright 2015 Zhu.Jin Liang


#ifndef CAFFE_UTIL_COORDS_H_
#define CAFFE_UTIL_COORDS_H_

#include <algorithm>
#include <utility>
#include <vector>

#include "caffe/util/util_pre_define.hpp"

namespace caffe {

template <typename Dtype> class Net;

template <typename Dtype>
class DiagonalAffineMap {
 public:
  explicit DiagonalAffineMap(const vector<pair<Dtype, Dtype> > coefs)
    : coefs_(coefs) { }
  static DiagonalAffineMap identity(const int nd) {
    return DiagonalAffineMap(vector<pair<Dtype, Dtype> >(nd, make_pair(1, 0)));
  }

  inline DiagonalAffineMap compose(const DiagonalAffineMap& other) const {
    DiagonalAffineMap<Dtype> out;
    if (coefs_.size() == other.coefs_.size()) {
      transform(coefs_.begin(), coefs_.end(), other.coefs_.begin(),
          std::back_inserter(out.coefs_), &compose_coefs);
    } else {
      // 为了支持CropPatchFromMaxFeaturePositionLayer
      if ( (coefs_.size() == 2) && (other.coefs_.size() % coefs_.size() == 0) ) {

        for (int i = 0; i < other.coefs_.size(); i += 2) {
          out.coefs_.push_back(compose_coefs(coefs_[0], other.coefs_[i]));
          out.coefs_.push_back(compose_coefs(coefs_[1], other.coefs_[i + 1]));
        }

      } else if ( (other.coefs_.size() == 2) && (coefs_.size() % other.coefs_.size() == 0) ) {

        for (int i = 0; i < coefs_.size(); i += 2) {
          out.coefs_.push_back(compose_coefs(coefs_[i], other.coefs_[0]));
          out.coefs_.push_back(compose_coefs(coefs_[i + 1], other.coefs_[1]));
        }

      } else {
       LOG(FATAL) << "Attempt to compose DiagonalAffineMaps of different dimensions: "
         << coefs_.size() << " vs " << other.coefs_.size();
      }
    }
    // 判断所有的coefs_是否相等，如果是，只返回一个
    if (out.coefs_.size() > 2 && out.coefs_.size() % 2 == 0) {
			bool isOK = true;
			for (int i = 2; i < out.coefs_.size() && isOK; i += 2) {
				isOK = IsEqual(out.coefs_[0].first, out.coefs_[i].first)
						&& IsEqual(out.coefs_[0].second, out.coefs_[i].second)
						&& IsEqual(out.coefs_[1].first, out.coefs_[i + 1].first)
						&& IsEqual(out.coefs_[1].second, out.coefs_[i + 1].second);
			}
			if (isOK) {
				out.coefs_.erase(out.coefs_.begin() + 2, out.coefs_.end());
			}
    }
    return out;
  }
  inline DiagonalAffineMap inv() const {
    DiagonalAffineMap<Dtype> out;
    transform(coefs_.begin(), coefs_.end(), std::back_inserter(out.coefs_),
        &inv_coefs);
    return out;
  }
  inline vector<pair<Dtype, Dtype> > coefs() { return coefs_; }

 private:
  DiagonalAffineMap() { }
  static inline pair<Dtype, Dtype> compose_coefs(pair<Dtype, Dtype> left,
      pair<Dtype, Dtype> right) {
    return make_pair(left.first * right.first,
                     left.first * right.second + left.second);
  }
  static inline pair<Dtype, Dtype> inv_coefs(pair<Dtype, Dtype> coefs) {
    return make_pair(1 / coefs.first, - coefs.second / coefs.first);
  }
  vector<pair<Dtype, Dtype> > coefs_;
};

template <typename Dtype>
DiagonalAffineMap<Dtype> FilterMap(const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w, const int pad_h, const int pad_w) {
  vector<pair<Dtype, Dtype> > coefs;
  coefs.push_back(make_pair(stride_h,
        static_cast<Dtype>(kernel_h - 1) / 2 - pad_h));
  coefs.push_back(make_pair(stride_w,
        static_cast<Dtype>(kernel_w - 1) / 2 - pad_w));

  return DiagonalAffineMap<Dtype>(coefs);
}

}  // namespace caffe

#endif  // CAFFE_UTIL_COORDS_H_