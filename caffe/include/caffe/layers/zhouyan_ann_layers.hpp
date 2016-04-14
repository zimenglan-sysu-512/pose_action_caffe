  // Copyright 2016 DDK

#ifndef CAFFE_ZHOUYAN_ANN_LAYERS_HPP_
#define CAFFE_ZHOUYAN_ANN_LAYERS_HPP_

#include <string>
#include <utility>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "hdf5.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/net.hpp"
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/data_layers.hpp"
#include "caffe/loss_layers.hpp"
#include "caffe/internal_thread.hpp"

using std::vector;
using std::string;

namespace caffe {

template <typename Dtype>
class LoadDataFromFileLayer : public BasePrefetchingDataLayer<Dtype> 
{
 public:
  explicit LoadDataFromFileLayer(
    const LayerParameter& param)
      : BasePrefetchingDataLayer<Dtype>(param) {}
  virtual ~LoadDataFromFileLayer();
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { 
    return "LoadDataFromFile"; 
  }

  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 2; }

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);

 protected:
  virtual void ShuffleImages();
  virtual void InternalThreadEntry();
  virtual void ReadDataFromFile(Blob<Dtype>* blob, const int n, 
                                const int c, const std::string path);
  virtual void VisualDataFromBlob(Blob<Dtype>* blob, const int n, 
                                  const int c, const std::string path);
  Dtype scale_;
  bool shuffle_;
  int lines_id_;
  int batch_size_;
  int ow_, oh_, on_;
  std::string source_;
  std::string root_folder_;
  std::string visual_path_;
  // variables -- global
  std::vector<std::string> objidxs_;
  std::vector<std::string> imgidxs_;
  std::vector<std::string> images_paths_;
  // <<f1, f2, f3, [f4]>>
  std::vector<std::vector<std::string> >lines_;

  shared_ptr<Caffe::RNG> rng_;
  shared_ptr<Caffe::RNG> prefetch_rng_;
};

}  // namespace caffe

#endif  // CAFFE_ZHOUYAN_ANN_LAYERS_HPP_