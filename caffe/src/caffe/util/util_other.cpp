// CopyRight 2015 ZhuJin Liang

#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/io/coded_stream.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

#include <stdio.h>

#include "caffe/common.hpp"
#include "caffe/util/util_others.hpp"
#include "caffe/util/util_pre_define.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {


/// Generate bounding boxes from corresponding coordinates
void GetBBoxes(const vector<float>& coords, const int key_points_count,
		vector<vector<float> >& bboxes) 
{
	const int clen = coords.size();
	const int ilen = clen % (key_points_count * 2);
	const int blen = clen / (key_points_count * 2);
	CHECK(ilen == 0) << "The number of key points is wrong.";
	bboxes = vector<vector<float> >(blen, vector<float>(4, -1));
	/// get bbox
	for (int i = 0; i < bboxes.size(); ++i) {
		float min_x = -1, max_x = -1;
		float min_y = -1, max_y = -1;
		for (int j = 0; j < key_points_count; ++j) {
			int idx = i * key_points_count + j;
			idx *= 2;
			if (std::abs(coords[idx] - (-1)) < ELLISION
					|| std::abs(coords[idx + 1] - (-1)) < ELLISION) {
				continue;
			}

			if (std::abs(min_x - (-1)) < ELLISION) {
				min_x = coords[idx];
				max_x = coords[idx];

				min_y = coords[idx + 1];
				max_y = coords[idx + 1];
			} else {
				min_x = MIN(min_x, coords[idx]);
				max_x = MAX(max_x, coords[idx]);

				min_y = MIN(min_y, coords[idx + 1]);
				max_y = MAX(max_y, coords[idx + 1]);
			}
		}
		bboxes[i][0] = min_x;
		bboxes[i][1] = min_y;

		bboxes[i][2] = max_x;
		bboxes[i][3] = max_y;
	}
}

/// First compute the diagonal length from each bbox that generated by calling 
/// GetBBoxes function, and then compute the ratio from 
/// standard_bbox_diagonal_len variable.
/// Return the array of ratio of each bbox.
void GetBBoxStandardScale(const vector<float>& coords,
		const int key_points_count, 
		const int standard_bbox_diagonal_len,
		vector<float>& standard_scale) 
{
	standard_scale.clear();
	vector<vector<float> > bboxes;
	GetBBoxes(coords, key_points_count, bboxes);

	for (int j = 0; j < bboxes.size(); ++j) {
		if (std::abs(bboxes[j][0] - (-1)) < ELLISION) {
			standard_scale.push_back(1);
		} else {
			float w = bboxes[j][0] - bboxes[j][2];
			float h = bboxes[j][1] - bboxes[j][3];
			float d = std::sqrt(w * w + h * h);
			float s = standard_bbox_diagonal_len / d;
			standard_scale.push_back(s);
		}
	}
}

/// Get array of ratio of each bbox of each sample.
void GetAllBBoxStandardScale(
		const vector<std::pair<std::string, vector<float> > >& samples,
		const int key_points_count, const int standard_bbox_diagonal_len,
		vector<vector<float> >& bboxes_standard_scale) 
{
	bboxes_standard_scale = vector<vector<float> >(
			samples.size(), vector<float>());
	//
	for (int i = 0; i < samples.size(); ++i) {
		GetBBoxStandardScale(samples[i].second, key_points_count,
				standard_bbox_diagonal_len, bboxes_standard_scale[i]);
	}
}

template<typename Dtype>
bool compareCandidate(const pair<Dtype, vector<float> >& c1,
		const pair<Dtype, vector<float> >& c2) 
{
	return c1.first >= c2.first;
}
template bool compareCandidate<float>(const pair<float, vector<float> >& c1,
		const pair<float, vector<float> >& c2);
template bool compareCandidate<double>(const pair<double, vector<float> >& c1,
		const pair<double, vector<float> >& c2);

// ################################################################################
/// NMS
template<typename Dtype>
const vector<bool> nms(vector<pair<Dtype, vector<float> > >& candidates,
		const float overlap, const int top_N, const bool addScore) 
{
	vector<bool> mask(candidates.size(), false);
	if (mask.size() == 0)
		return mask;
	/// sort
	vector<bool> skip(candidates.size(), false);
	std::stable_sort(candidates.begin(), candidates.end(), 
			compareCandidate<Dtype>);
	/// get area
	vector<float> areas(candidates.size(), 0);
	for (int i = 0; i < candidates.size(); ++i) {
		areas[i] = (candidates[i].second[2] - candidates[i].second[0] + 1)
				* (candidates[i].second[3] - candidates[i].second[1] + 1);
	}
	for (int count = 0, i = 0; count < top_N && i < mask.size(); ++i) {
		if (skip[i])
			continue;
		// set count & mask[i]
		mask[i] = true;
		++count;
		// suppress the significantly covered bbox
		for (int j = i + 1; j < mask.size(); ++j) {
			if (skip[j])
				continue;
			// get intersections
			float xx1 = MAX(candidates[i].second[0], candidates[j].second[0]);
			float yy1 = MAX(candidates[i].second[1], candidates[j].second[1]);
			float xx2 = MIN(candidates[i].second[2], candidates[j].second[2]);
			float yy2 = MIN(candidates[i].second[3], candidates[j].second[3]);
			float w = xx2 - xx1 + 1;
			float h = yy2 - yy1 + 1;
			if (w > 0 && h > 0) {
				// compute overlap
				float o = w * h / areas[j];
				if (o > overlap) {
					skip[j] = true;
					// add scores
					if (addScore) {
						candidates[i].first += candidates[j].first;
					}
				}
			}
		}
	}

	return mask;
}
template const vector<bool> nms<float>(
		vector<pair<float, vector<float> > >& candidates, 
		const float overlap,
		const int top_N, const bool addScore = false);
template const vector<bool> nms<double>(
		vector<pair<double, vector<float> > >& candidates, 
		const float overlap,
		const int top_N, const bool addScore = false);

template<typename Dtype>
Dtype GetArea(const vector<Dtype>& bbox) {
	Dtype w = bbox[2] - bbox[0] + 1;
	Dtype h = bbox[3] - bbox[1] + 1;
	if (w <= 0 || h <= 0)
		return Dtype(0.0);
	return w * h;
}
template float GetArea(const vector<float>& bbox);
template double GetArea(const vector<double>& bbox);

template<typename Dtype>
Dtype GetArea(const Dtype x1, const Dtype y1, 
		const Dtype x2, const Dtype y2) 
{
	Dtype w = x2 - x1 + 1;
	Dtype h = y2 - y1 + 1;
	if (w <= 0 || h <= 0)
		return Dtype(0.0);
	return w * h;
}
template float GetArea(const float x1, const float y1, 
		const float x2, const float y2);
template double GetArea(const double x1, const double y1, 
		const double x2, const double y2);

template<typename Dtype>
Dtype GetOverlap(const vector<Dtype>& bbox1, 
		const vector<Dtype>& bbox2) 
{
	Dtype x1 = MAX(bbox1[0], bbox2[0]);
	Dtype y1 = MAX(bbox1[1], bbox2[1]);
	Dtype x2 = MIN(bbox1[2], bbox2[2]);
	Dtype y2 = MIN(bbox1[3], bbox2[3]);
	Dtype w = x2 - x1 + 1;
	Dtype h = y2 - y1 + 1;
	if (w <= 0 || h <= 0)
		return Dtype(0.0);

	Dtype intersection = w * h;
	Dtype area1 = GetArea(bbox1);
	Dtype area2 = GetArea(bbox2);
	Dtype u = area1 + area2 - intersection;

	return intersection / u;
}
template float GetOverlap(const vector<float>& bbox1,
		const vector<float>& bbox2);
template double GetOverlap(const vector<double>& bbox1,
		const vector<double>& bbox2);

template<typename Dtype>
Dtype GetOverlap(
		const Dtype x11, const Dtype y11, 
		const Dtype x12, const Dtype y12,
		const Dtype x21, const Dtype y21, 
		const Dtype x22, const Dtype y22) 
{
	Dtype x1 = MAX(x11, x21);
	Dtype y1 = MAX(y11, y21);
	Dtype x2 = MIN(x12, x22);
	Dtype y2 = MIN(y12, y22);
	Dtype w = x2 - x1 + 1;
	Dtype h = y2 - y1 + 1;
	if (w <= 0 || h <= 0)
		return Dtype(0.0);

	Dtype intersection = w * h;
	Dtype area1 = GetArea(x11, y11, x12, y12);
	Dtype area2 = GetArea(x21, y21, x22, y22);
	Dtype u = area1 + area2 - intersection;

	return intersection / u;
}
template float GetOverlap(
		const float x11, const float y11, 
		const float x12, const float y12,
		const float x21, const float y21, 
		const float x22, const float y22);
template double GetOverlap(
		const double x11, const double y11, 
		const double x12, const double y12,
		const double x21, const double y21, 
		const double x22, const double y22);

template<typename Dtype>
Dtype GetNMSOverlap(const vector<Dtype>& bbox1, 
		const vector<Dtype>& bbox2) 
{
	Dtype x1 = MAX(bbox1[0], bbox2[0]);
	Dtype y1 = MAX(bbox1[1], bbox2[1]);
	Dtype x2 = MIN(bbox1[2], bbox2[2]);
	Dtype y2 = MIN(bbox1[3], bbox2[3]);
	Dtype w = x2 - x1 + 1;
	Dtype h = y2 - y1 + 1;
	if (w <= 0 || h <= 0)
		return 0.0;

	Dtype area2 = GetArea(bbox2);
	return w * h / area2;
}
template float GetNMSOverlap(const vector<float>& bbox1,
		const vector<float>& bbox2);
template double GetNMSOverlap(const vector<double>& bbox1,
		const vector<double>& bbox2);

/// bbox1: x11, y11, x12, y12
/// bbox2: x21, y21, x22, y22
template<typename Dtype>
Dtype GetNMSOverlap(
		const Dtype x11, const Dtype y11, 
		const Dtype x12, const Dtype y12,
		const Dtype x21, const Dtype y21, 
		const Dtype x22, const Dtype y22) 
{
	Dtype x1 = MAX(x11, x21);
	Dtype y1 = MAX(y11, y21);
	Dtype x2 = MIN(x12, x22);
	Dtype y2 = MIN(y12, y22);
	Dtype w = x2 - x1 + 1;
	Dtype h = y2 - y1 + 1;
	if (w <= 0 || h <= 0)
		return Dtype(0.0);

	Dtype area2 = GetArea(x21, y21, x22, y22);
	return w * h / area2;
}
template float GetNMSOverlap(
		const float x11, const float y11, 
		const float x12, const float y12,
		const float x21, const float y21, 
		const float x22, const float y22);
template double GetNMSOverlap(
		const double x11, const double y11, 
		const double x12, const double y12,
		const double x21, const double y21, 
		const double x22, const double y22);

// ################################################################################
/// Face
vector<bool> GetPredictedResult(
		const vector<std::pair<int, vector<float> > > &gt_instances,
		const vector<std::pair<float, vector<float> > > &pred_instances,
		float ratio) 
{
	vector<bool> res;
	vector<bool> used_gt_instance;
	used_gt_instance.resize(gt_instances.size(), false);
	for (int pred_id = 0; pred_id < pred_instances.size(); pred_id++) {
		float max_overlap = 0;
		int used_id = -1;
		for (int gt_id = 0; gt_id < gt_instances.size(); ++gt_id) {
			float overlap = GetOverlap(pred_instances[pred_id].second,
					gt_instances[gt_id].second);
			if (overlap > max_overlap) {
				max_overlap = overlap;
				used_id = gt_id;
			}
		}
		if (used_id != -1 && max_overlap >= ratio 
					&& used_gt_instance[used_id] == false) {
			res.push_back(true);
			used_gt_instance[used_id] = true;
		} else {
			res.push_back(false);
		}
	}
	return res;
}

float GetTPFPPoint_FDDB(
		vector<std::pair<float, vector<float> > >& pred_instances_with_gt,
		const int n_positive, vector<float>& tpr, vector<float> &fpr) 
{
	std::stable_sort(pred_instances_with_gt.begin(), pred_instances_with_gt.end(),
			compareCandidate<float>);
	tpr.clear();
	fpr.clear();
	int tp = 0;
	int negative_count = 0;
	for (int i = 0; i < pred_instances_with_gt.size(); i++) {
		negative_count += int(pred_instances_with_gt[i].second[4]) == 0 ? 1 : 0;
	}

	for (int i = 0; i < pred_instances_with_gt.size(); i++) {
		tp += int(pred_instances_with_gt[i].second[4]) == 1 ? 1 : 0;
		fpr.push_back((i + 1 - tp) / (0.0 + negative_count));
		tpr.push_back(tp / (0.0 + n_positive));

	}
	float auc = tpr[0] * fpr[0];
	for (int i = 1; i < pred_instances_with_gt.size(); i++) {
		auc += tpr[i] * (fpr[i] - fpr[i - 1]);
	}
	return auc;
}

float GetPRPoint_FDDB(
		vector<std::pair<float, vector<float> > >& pred_instances_with_gt,
		const int n_positive, vector<float>& precision, vector<float> &recall) 
{

	std::stable_sort(pred_instances_with_gt.begin(), pred_instances_with_gt.end(),
			compareCandidate<float>);
	precision.clear();
	recall.clear();
	int corrected_count = 0;
	for (int i = 0; i < pred_instances_with_gt.size(); i++) {
		corrected_count += int(pred_instances_with_gt[i].second[4]) == 1 ? 1 : 0;
		precision.push_back(corrected_count / (i + 0.0 + 1));
		recall.push_back(corrected_count / (0.0 + n_positive));

	}
	float ap = precision[0] * recall[0];
	for (int i = 1; i < pred_instances_with_gt.size(); i++) {
		ap += precision[i] * (recall[i] - recall[i - 1]);
	}
	return ap;
}

void GetPredictedWithGT_FDDB(const string gt_file, const string pred_file,
		vector<std::pair<float, vector<float> > >& pred_instances_with_gt,
		int & n_positive, bool showing, string img_folder, string output_folder,
		float ratio) 
{
	FILE* gt_fd = NULL;
	FILE* pred_fd = NULL;
	gt_fd = fopen(gt_file.c_str(), "r");
	CHECK(gt_fd != NULL) << " can not find gt_file " << gt_file;
	pred_fd = fopen(pred_file.c_str(), "r");
	CHECK(pred_fd != NULL) << " can not find pred_file " << pred_file;
	char img_name[255];
	char pred_img_name[255];
	n_positive = 0;
	pred_instances_with_gt.clear();
	cv::Mat src_img;
	char scores_c[100];
	while (fscanf(gt_fd, "%s", img_name) == 1) {
		if (showing) {

			src_img = cv::imread(img_folder + string(img_name) + string(".jpg"),
					CV_LOAD_IMAGE_COLOR);
			if (!src_img.data) {
				LOG(ERROR)<< "Could not open or find file " <<
				img_folder +string(img_name)+string(".jpg");
			}

		}

		CHECK(fscanf(pred_fd,"%s",pred_img_name) == 1 && strcmp(pred_img_name,img_name) == 0);
		int n_face = 0;
		CHECK(fscanf(gt_fd,"%d",&n_face) == 1);
		//LOG(INFO)<<"image:"<<img_folder +string(img_name)+string(".jpg") <<"has " <<n_face<<" face";

		vector< std::pair<int, vector<float> > > gt_instances;
		vector< std::pair<int, vector<float> > > gt_instances_nohard;
		vector< std::pair<float, vector<float> > > pred_instances;

		for(int i=0; i < n_face; i++)
		{
			/**
			 * read faces in one image for gt
			 */
			float lt_x, lt_y, height,width;
			int label;
			CHECK(fscanf(gt_fd, "%f %f %f %f %d",&lt_x, &lt_y, &width, &height, &label) == 5);
			vector<float> temp;
			temp.push_back(lt_x);
			temp.push_back(lt_y);
			temp.push_back(lt_x+width);
			temp.push_back(lt_y+height);
			gt_instances.push_back( std::make_pair(label, temp));
			if(label == 0)
			{
				gt_instances_nohard.push_back(std::make_pair(label, temp));
				if(showing)
				{
					cv::rectangle(src_img, cv::Point(temp[0], temp[1]),
							cv::Point(temp[2], temp[3]), cv::Scalar(0, 0, 255));
				}
				n_positive += 1;
			}
		}

		CHECK(fscanf(pred_fd,"%d",&n_face) == 1);
		for(int i=0; i < n_face; i++)
		{
			float lt_x, lt_y, height,width,score;
			/**
			 * read faces in one image for gred
			 */
			vector<float> temp;
			temp.clear();
			CHECK(fscanf(pred_fd, "%f %f %f %f %f",&lt_x, &lt_y, &width, &height, &score) == 5);
			temp.push_back(lt_x);
			temp.push_back(lt_y);
			temp.push_back(lt_x+width);
			temp.push_back(lt_y+height);
			pred_instances.push_back(std::make_pair(score,temp));

			if(showing)
			{
				cv::rectangle(src_img, cv::Point(temp[0], temp[1]),
						cv::Point(temp[2], temp[3]), cv::Scalar(255, 0, 0));
				sprintf(scores_c, "%.3f", score);
				cv::putText(src_img, scores_c, cv::Point(temp[0]/2 + temp[2]/2, temp[1]),
						CV_FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 255, 255));
			}
		}
		// if(showing)
		// {
		// 	LOG(INFO)<<"saving :"<<output_folder +string(img_name)+string(".jpg");
		// 	cv::imwrite(output_folder +string(img_name)+string(".jpg"), src_img);
		// }

		vector<bool> corrected = GetPredictedResult(gt_instances,pred_instances,ratio);
		vector<bool> corrected_nohard = GetPredictedResult(gt_instances_nohard,pred_instances,ratio );
		for(int i=0; i < n_face; i++)
		{
			if( corrected[i] != corrected_nohard[i])
			{
				LOG(INFO)<<"Detected a hard sample";
				continue;
			}
			pred_instances[i].second.push_back( corrected[i] == true? 1:0);
			pred_instances_with_gt.push_back(pred_instances[i]);
		}
		{
			vector<bool> used_gt_instance;
			used_gt_instance.resize(gt_instances.size(), false);
			for (int pred_id = 0; pred_id < pred_instances.size(); pred_id++) {
				float max_overlap = 0;
				int used_id = -1;
				for (int gt_id = 0; gt_id < gt_instances.size(); ++gt_id) {
					float overlap = GetOverlap(pred_instances[pred_id].second,
							gt_instances[gt_id].second);
					if (overlap > max_overlap) {
						max_overlap = overlap;
						used_id = gt_id;
					}
				}
				if (used_id != -1) {
					used_gt_instance[used_id] = true;
				}
			}
			bool allCorrect = true;
			for(int gt_id = 0; gt_id < used_gt_instance.size() && allCorrect; ++gt_id) {
				allCorrect = used_gt_instance[gt_id];
			}
			if(!allCorrect)
			{
				LOG(INFO)<<"saving :"<<output_folder +string(img_name)+string(".jpg");
				cv::imwrite(output_folder +string(img_name)+string(".jpg"), src_img);
			}
		}
	}

	fclose(pred_fd);
	fclose(gt_fd);
}

}
// namespace caffe
