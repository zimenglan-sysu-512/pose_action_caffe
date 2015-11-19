#ifndef CAFFE_OPTIMIZATION_SOLVER2_HPP_
#define CAFFE_OPTIMIZATION_SOLVER2_HPP_

#include <opencv2/core/core.hpp>

#include <string>
#include <vector>
#include <utility>
#include <map>
#include <queue>
#include <deque>

#include "caffe/net.hpp"
#include "caffe/solver.hpp"
#include "caffe/vision_layers.hpp"

using std::string;
using std::pair;
using std::make_pair;
using std::map;
using std::vector;
using std::deque;
using cv::Point_;

namespace caffe {

enum GeometrySGDSolver_TrainPhase {
  KEY_POINT = 0, PATCH = 1, BBOX = 2
};

const static std::string GeometrySGDSolver_TrainPhase_String[3] = 
    {"KEY_POINT", "PATCH", "BBOX"};

/**
 * @brief An interface for classes that perform optimization on Net%s.
 *
 * Requires implementation of ComputeUpdateValue to compute a parameter update
 * given the current state of the Net parameters.
 */
/**
 * @brief Optimizes the parameters of a Net using
 *        stochastic gradient descent (SGD) with momentum.
 */
template <typename Dtype>
class GeoSGDSolver : public Solver<Dtype> {
 public:
  explicit GeoSGDSolver(const SolverParameter& param)
      : Solver<Dtype>(param) { PreSolve(); }
  explicit GeoSGDSolver(const string& param_file)
      : Solver<Dtype>(param_file) { PreSolve(); }

  const vector<shared_ptr<Blob<Dtype> > >& history() {
    return history_; 
  }
  Dtype LearningRate() { return GetLearningRate(); }
  ///
  virtual void Do_Test();

 protected:
  Dtype GetLearningRate();
  void PreSolve();
  virtual void ComputeUpdateValue();
  virtual void ClipGradients();
  virtual void SnapshotSolverState(SolverState * state);
  virtual void RestoreSolverState(const SolverState& state);
  // history maintains the historical momentum data.
  // update maintains update related data and is not needed in snapshots.
  // temp maintains other information that might be needed in computation
  //   of gradients/updates and is not needed in snapshots
  vector<shared_ptr<Blob<Dtype> > > history_;
  vector<shared_ptr<Blob<Dtype> > > update_;
  vector<shared_ptr<Blob<Dtype> > > temp_;

  ///
  virtual void bootstrap() {
    bootstrap(trainPhase_);
  }
  void bootstrap(const GeometrySGDSolver_TrainPhase& trainPhase);

  // The test routine
  virtual void Test();

  GeoSGDSolverParameter geometry_param_;
  vector<SolverParameter> solver_params;
  GeometrySGDSolver_TrainPhase trainPhase_;

  // history maintains the historical momentum data.
  // vector<shared_ptr<Blob<Dtype> > > history_;

  map<int, int> patch_idxs;

  string imgs_folder_;
  vector<pair<string, vector<float> > > test_samples_;
  vector<pair<string, vector<int> > > test_samples_tags_;

  vector<int> key_point_idxs_;
  vector<int> key_point_idxs_patch_bbox_;
  vector<float> pyramid_scales_;

  bool test_only;
  int valid_distance_, invalid_distance_;
  int key_point_counts_;
  int standard_bbox_diagonal_len_;
  float ratio_beg_, ratio_end_, ratio_factor_;
  float scale_upper_limit_, scale_lower_limit_;
  float standard_image_len_;
  float heat_map_a_, heat_map_b_;

 private:
  struct HEAT_MAP_INFO {
    shared_ptr<Blob<Dtype> > heat_map;
    shared_ptr<Blob<Dtype> > bboxes;
    int w_c;
    int h_c;
    cv::Mat img;
    vector<float> coords;
    float pyramid_scale;
    // scale from original
    float scale;
    string filename;

    HEAT_MAP_INFO(){}

    HEAT_MAP_INFO(const HEAT_MAP_INFO& heat_map_info) {
      heat_map = heat_map_info.heat_map;
      bboxes = heat_map_info.bboxes;
      w_c = heat_map_info.w_c;
      h_c = heat_map_info.h_c;
      img = heat_map_info.img.clone();
      coords = heat_map_info.coords;
      pyramid_scale = heat_map_info.pyramid_scale;
      scale = heat_map_info.scale;
      filename = heat_map_info.filename;
    }

    HEAT_MAP_INFO& operator=(const HEAT_MAP_INFO& heat_map_info) {
      heat_map = heat_map_info.heat_map;
      bboxes = heat_map_info.bboxes;
      w_c = heat_map_info.w_c;
      h_c = heat_map_info.h_c;
      img = heat_map_info.img.clone();
      coords = heat_map_info.coords;
      pyramid_scale = heat_map_info.pyramid_scale;
      scale = heat_map_info.scale;
      filename = heat_map_info.filename;

      return *this;
    }
  };

  struct FINAL_HEAT_MAP_INFO {
    Blob<Dtype> heat_map;
    Blob<float> scales;
  };
  //
  void SetMode(GeometrySGDSolver_TrainPhase phase);
  //
  bool Train_Forward(const vector<Blob<Dtype>*>& bottom_vec);

  // since one input image can split into several images,
  // in order to deal with large scale image
  // one input image can produce several images, 
  // this function concat sub heatmap
  void ConcatHeatMap(const vector<Blob<Dtype>* >& key_point_input,
      const vector<Blob<Dtype>* >& key_point_output, 
      std::deque<HEAT_MAP_INFO>& heat_maps_done,
      std::deque<HEAT_MAP_INFO>& heat_maps_undone, 
      const char* output_path = "cache/GeometrySGDSolver/KEY_POINT");

  // get the pyramid heat map from a given image, 
  // which is recored in variable pyramid_heat_maps
  // the return value the the given image path(sample.first)
  void GetPyramidHeatMap(
      const pair<string, vector<float> >& sample, 
      const vector<float>& pyramid_scales,
      Net<Dtype>* heat_map_net, 
      deque<HEAT_MAP_INFO>& pyramid_heat_maps, 
      cv::Mat& src_img);

  void ResizeImageWithBlank(const cv::Mat& src, 
      cv::Mat& dst, const float scale);

  // Get local maximum along channels
  // 4 neighborhood
  void GetNDimLocalMaximum(const Blob<Dtype>& heat_maps, 
      Blob<Dtype>& final_heat_map,
      vector<vector<vector<float> > >& mask_with_scale, 
      const vector<float>& scales,
      FINAL_HEAT_MAP_INFO* final_heat_map_max_info = NULL);

  // if mask[c][h][w] is not equal to -1,
  // it means that there is a local maximum in point(x, y) 
  // in for key point c(index)
  // and the value mask.data_at(0, c, h, w) is 
  // the scales from original image to pyramid images
  void GetSampleFinalHeatmap(const pair<string, vector<float> >& sample,
      const vector<float>& pyramid_scales, 
      Net<Dtype>* heat_map_net, 
      Blob<Dtype>& final_heat_map,
      vector<vector<vector<float> > >& mask_with_scale, 
      cv::Mat& img, 
      FINAL_HEAT_MAP_INFO* final_heat_map_max_info = NULL);
  //
  void Test_HeatMapNet();

  DISABLE_COPY_AND_ASSIGN(GeoSGDSolver);
};

}  // namespace caffe

#endif  // CAFFE_OPTIMIZATION_SOLVER2_HPP_
