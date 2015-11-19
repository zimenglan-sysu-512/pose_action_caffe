// Copyright 2015 Zhu.Jin Liang

#include "caffe/common.hpp"
#include "caffe/util/coords.hpp"
#include "caffe/util/util_coords.hpp"
#include "caffe/util/util_pre_define.hpp"

namespace caffe {

template <typename Dtype>
DiagonalAffineMap<Dtype> GetMapBetweenFeatureMap(
    const vector<Blob<Dtype>*>& bottom,
		const Net<Dtype>* const net) 
{
	CHECK_GE(bottom.size(), 2) << "At least two input blob";

  // Construct a map from top blobs to layer inds, skipping over in-place
  // connections.
  map<Blob<Dtype>*, int> down_map;
  for (int layer_ind = 0; layer_ind < net->top_vecs().size();
       ++layer_ind) {
    vector<Blob<Dtype>*> tops = net->top_vecs()[layer_ind];
    for (int top_ind = 0; top_ind < tops.size(); ++top_ind) {
      if (down_map.find(tops[top_ind]) == down_map.end()) {
        down_map[tops[top_ind]] = layer_ind;
      }
    }
  }
  // Walk back from the first bottom, keeping track of all the blobs we pass.
  set<Blob<Dtype>*> path_blobs;
  Blob<Dtype>* blob = bottom[0];
  int layer_ind;
  // TODO this logic can be simplified if all blobs are tops
  path_blobs.insert(blob);
  while (down_map.find(blob) != down_map.end()) {
    layer_ind = down_map[blob];
    if (net->bottom_vecs()[layer_ind].size() == 0) {
      break;
    }
    blob = net->bottom_vecs()[layer_ind][0];
    path_blobs.insert(blob);
  }
  // Now walk back from the second bottom, until we find a blob of intersection.
  Blob<Dtype>* inter_blob = bottom[1];
  while (path_blobs.find(inter_blob) == path_blobs.end()) {
    CHECK(down_map.find(inter_blob) != down_map.end())
        << "Cannot align apparently disconnected blobs.";
    layer_ind = down_map[inter_blob];
    CHECK_GT(net->bottom_vecs()[layer_ind].size(), 0)
        << "Cannot align apparently disconnected blobs.";
    inter_blob = net->bottom_vecs()[layer_ind][0];
  }
  // Compute the coord map from the blob of intersection to each bottom.
  vector<DiagonalAffineMap<Dtype> > coord_maps(2,
      DiagonalAffineMap<Dtype>::identity(2));
  for (int i = 0; i < 2; ++i) {
    for (Blob<Dtype>* blob = bottom[i]; blob != inter_blob;
         blob = net->bottom_vecs()[down_map[blob]][0]) {
      shared_ptr<Layer<Dtype> > layer = net->layers()[down_map[blob]];
      // LOG(INFO);
      // LOG(INFO) << "layer -- name: " << layer->type();
      // LOG(INFO);
      coord_maps[i] = coord_maps[i].compose(layer->coord_map());
    }
  }
  // Compute the mapping from first bottom coordinates to second.
  return coord_maps[1].compose(coord_maps[0].inv());
}

template DiagonalAffineMap<float> GetMapBetweenFeatureMap(
    const vector<Blob<float>*>& bottom,
		const Net<float>* const net);
template DiagonalAffineMap<double> GetMapBetweenFeatureMap(
    const vector<Blob<double>*>& bottom,
		const Net<double>* const net);

} // namespace caffe