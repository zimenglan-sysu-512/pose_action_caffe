#ifndef CAFFE_UTIL_UTIL_OTHER_H_
#define CAFFE_UTIL_UTIL_OTHER_H_

#include <vector>
#include <map>
#include <utility>
using std::vector;
using std::pair;
using std::string;

namespace caffe {

/*
 * Return the bboxes(each is 4 dims, x1, y1, x2, y2) of the given coords
 * If no bbox found(all the annotations are -1), then return four -1
 */
void GetBBoxes(const vector<float>& coords, 
		const int key_points_count, vector<vector<float> >& bboxes);

/*
 * scale = standard_len / actual_len
 */
void GetBBoxStandardScale(const vector<float>& coords, const int key_points_count,
		const int standard_bbox_diagonal_len, vector<float>& standard_scale);

void GetAllBBoxStandardScale(
		const vector<std::pair<std::string, vector<float> > >& samples,
		const int key_points_count, const int standard_bbox_diagonal_len,
		vector<vector<float> >& bboxes_standard_scale);

template <typename Dtype>
bool compareCandidate(const pair<Dtype, vector<float> >& c1,
    const pair<Dtype, vector<float> >& c2);

// Non-maximum suppression. return a mask which elements are selected
//   overlap   Overlap threshold for suppression
//             For a selected box Bi, all boxes Bj that are covered by
//             more than overlap are suppressed. Note that 'covered' is
//             is |Bi \cap Bj| / |Bj|, not the PASCAL intersection over
//             union measure.
// if addscore == true, then the scores of all the overlap bboxes will be added
template <typename Dtype>
const vector<bool> nms(vector<pair<Dtype, vector<float> > >& candidates,
    const float overlap, const int top_N, const bool addScore = false);

template <typename Dtype>
Dtype GetArea(const vector<Dtype>& bbox);
template <typename Dtype>
Dtype GetArea(const Dtype x1, const Dtype y1, const Dtype x2, const Dtype y2);

// intersection over union
template <typename Dtype>
Dtype GetOverlap(const vector<Dtype>& bbox1, const vector<Dtype>& bbox2);
template <typename Dtype>
Dtype GetOverlap(const Dtype x11, const Dtype y11, const Dtype x12, const Dtype y12,
		const Dtype x21, const Dtype y21, const Dtype x22, const Dtype y22);

// |bbox1 \cap bbox2| / |bbox2|
template <typename Dtype>
Dtype GetNMSOverlap(const vector<Dtype>& bbox1, const vector<Dtype>& bbox2);
template <typename Dtype>
Dtype GetNMSOverlap(const Dtype x11, const Dtype y11, const Dtype x12, const Dtype y12,
		const Dtype x21, const Dtype y21, const Dtype x22, const Dtype y22);

vector<bool> GetPredictedResult(const vector< std::pair<int, vector<float> > > &gt_instances,
		const vector< std::pair<float, vector<float> > > &pred_instances, float ratio = 0.5);

float GetTPFPPoint_FDDB(vector< std::pair<float, vector<float> > >& pred_instances_with_gt,
		const int n_positive, vector<float>& tpr,vector<float> &fpr);

float GetPRPoint_FDDB(vector< std::pair<float, vector<float> > >& pred_instances_with_gt,
		const int n_positive, vector<float>& precision,vector<float> &recall);

void GetPredictedWithGT_FDDB(const string gt_file, const string pred_file,
		vector< std::pair<float, vector<float> > >& pred_instances_with_gt,
		int & n_positive, bool showing, string img_folder, string output_folder,float ratio = 0.5);

}  // namespace caffe

#endif   // CAFFE_UTIL_UTIL_OTHER_H_
