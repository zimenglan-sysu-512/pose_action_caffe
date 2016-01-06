// Copyright 2015 Zhu.Jin Liang

#include <algorithm>
#include <map>
#include <set>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/net.hpp"

#include "caffe/zhu_face_layers.hpp"
#include "caffe/util/util_coords.hpp"
#include "caffe/util/util_pre_define.hpp"

namespace caffe {

template <typename Dtype>
void CropPatchFromMaxFeatureFromTopKPositionsLayer<Dtype>::LayerSetUp(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
	// do nothing
}

template <typename Dtype>
void CropPatchFromMaxFeatureFromTopKPositionsLayer<Dtype>::Init(
			const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
	vector<pair<Dtype, Dtype> > coefs = 
			GetMapBetweenFeatureMap(bottom, this->net_).inv().coefs();
	coefs_.Reshape(1, 1, 2, 2);
	coefs_.mutable_cpu_data()[0] = coefs[0].first;
	coefs_.mutable_cpu_data()[1] = coefs[0].second;
	coefs_.mutable_cpu_data()[2] = coefs[1].first;
	coefs_.mutable_cpu_data()[3] = coefs[1].second;

	CHECK_EQ(bottom.size() - 2, top.size());
	CHECK_EQ(bottom[0]->num(),  bottom[1]->num());

	for (int i = 2; i < bottom.size(); ++i) {
		CHECK_EQ(bottom[0]->num(),    bottom[i]->num())
				<< "The num of bottom[2..n] shoud be equal to bottom[0]";
		CHECK_EQ(bottom[0]->width(),  bottom[i]->width())
				<< "The width of bottom[2..n] shoud be equal to bottom[0]";
		CHECK_EQ(bottom[0]->height(), bottom[i]->height())
				<< "The height of bottom[2..n] shoud be equal to bottom[0]";
	}

	CHECK(this->layer_param_.has_crop_patch_from_max_feature_position_param());
	const CropPatchFromMaxFeaturePositionParameter& crop_param = 
			this->layer_param_.crop_patch_from_max_feature_position_param();

	// 目前先固定死，截取的patch的是bottom[0]的 1/3
	const float scale = crop_param.crop_factor();
	crop_h_ = MAX(1, std::floor(bottom[0]->height() * scale));
	crop_w_ = MAX(1, std::floor(bottom[0]->width()  * scale));

	is_match_channel_.Reshape(top.size(), 1, 1, 1);
	int* is_match_channel = is_match_channel_.mutable_cpu_data();
	for (int i = 0; i < is_match_channel_.count(); ++i) {
		if (i < crop_param.match_channel_size() && crop_param.match_channel(i)) {
			is_match_channel[i] = 1;
		} else {
			is_match_channel[i] = 0;
		}
	}

	top_k_ = crop_param.top_k();
	nms_x_ = crop_param.nms_x();
	nms_y_ = crop_param.nms_y();
	CHECK_GT(top_k_, 0);
	CHECK_GE(nms_x_, 0);
	CHECK_GE(nms_y_, 0);
	// use default value
  if(this->layer_param_.is_disp_info()) {
  	LOG(INFO) << "layer name: "   << this->layer_param_.name()
  						<< " top_k: "			  << top_k_
  						<< " nms_x: "				<< nms_x_
  						<< " nms_y: "				<< nms_y_
  						<< " crop width: "  << crop_w_
  						<< " crop height: " << crop_h_;
  }
}

template <typename Dtype>
void CropPatchFromMaxFeatureFromTopKPositionsLayer<Dtype>::Reshape(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
	Init(bottom, top);
	const int* is_match_channel = is_match_channel_.cpu_data();
	for (int i = 0; i < top.size(); ++i) {
		if (is_match_channel[i] == 1) {
			CHECK_EQ(bottom[i + 2]->channels(), bottom[1]->channels());
			top[i]->Reshape(
					bottom[i]->num(), 
					bottom[i + 2]->channels() * top_k_, 
					crop_h_, 
					crop_w_
			);
		} else {
			top[i]->Reshape(
					bottom[i]->num(), 
					bottom[i + 2]->channels() * bottom[1]->channels() * top_k_,
					crop_h_, 
					crop_w_
			);
		}
	}
	crop_beg_.Reshape(
			bottom[1]->num(), 
			bottom[1]->channels() * top_k_, 
			1, 
			2
	);
}

// 先求出bottom[1] per channels最大值
template <typename Dtype>
void CropPatchFromMaxFeatureFromTopKPositionsLayer<Dtype>::DeriveCropBeg(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
	const Dtype* bottom1_data = bottom[1]->cpu_data();
	const Dtype* coefs 			  = coefs_.cpu_data();
	int* crop_beg 					  = crop_beg_.mutable_cpu_data();
	const int areas 					= bottom[1]->height() * bottom[1]->width();
	int idx1 		 = 0;
	int crop_idx = 0;
	for (int n = 0; n < bottom[1]->num(); ++n) {
		for (int c = 0; c < bottom[1]->channels(); ++c, crop_idx += 2) {

			Dtype max_val = bottom1_data[idx1];
			int max_idx = 0;

			for (int i = 0; i < areas; ++i, ++idx1) {
				if (bottom1_data[idx1] > max_val) {
					max_val = bottom1_data[idx1];
					max_idx = i;
				}
			}

			crop_beg[crop_idx + 0] = std::floor((coefs[0] * 
				(max_idx / bottom[1]->width()) + coefs[1])
					- crop_h_ / Dtype(2.));
			crop_beg[crop_idx + 1] = std::floor((coefs[2] * 
					(max_idx % bottom[1]->width()) + coefs[3])
					- crop_w_ / Dtype(2.));
		}
	}
}

// first find per channels top k highest score 
// and w.r.t. coordinates of the bottom[1] 
template <typename Dtype>
void CropPatchFromMaxFeatureFromTopKPositionsLayer<Dtype>::DeriveCropBegFromTopK(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
	int width  								= bottom[1]->width();
	int height 								= bottom[1]->height();
	const Dtype* coefs 				= coefs_.cpu_data();
	const Dtype* bottom1_data = bottom[1]->cpu_data();
	int* crop_beg 					  = crop_beg_.mutable_cpu_data();
	const int areas 					= width * height;

	int idx1      = 0;
	int crop_idx  = 0;
	int x1, y1, x2, y2;
	int bx1, by1, bx2, by2;	// valid

	for (int n = 0; n < bottom[1]->num(); ++n) {
		for (int c = 0; c < bottom[1]->channels(); ++c) {
			// <score, index>
			std::vector<std::pair<Dtype, int> > tops; 
			for (int i = 0; i < areas; ++i, ++idx1) {
			  tops.push_back(std::make_pair(bottom1_data[idx1], i));
			}
			// sort
			CHECK_EQ(areas, tops.size());
			std::sort(tops.begin(), tops.end(), std::greater<pair<Dtype, int> >());

			// find top k
			int k = 0;
			std::vector<bool> skips(areas, false);
			for(int i = 0; i < areas && k < this->top_k_; i++) {
				if(skips[i]) {
					continue;
				}
				k++;
				skips[i] = true;

				// get beg
				x1  = tops[i].second % width;
			  y1  = tops[i].second / width;
			  // set beg
			  crop_beg[crop_idx + 0] = std::floor((coefs[0] * y1 + coefs[1]) - crop_h_ / Dtype(2.));
			  crop_beg[crop_idx + 1] = std::floor((coefs[2] * x1 + coefs[3]) - crop_w_ / Dtype(2.));
			  crop_idx += 2;

			  // nms
			  if(k < top_k_ - 1) {
		  	  // nms square
		  	  bx1 = std::max(0, 				 x1 - nms_x_);
		  	  by1 = std::max(0, 				 y1 - nms_y_);
		  	  bx2 = std::min(width  - 1, x1 + nms_x_);
		  	  by2 = std::min(height - 1, y1 + nms_y_);
		  	  /// nms
		  	  int nms_offset;
		  	  for(int p = bx1; p <= bx2; p++) {
		  			for(int q = by1; q <= by2; q++) {
		  				nms_offset 				= q * width + p;
		  				skips[nms_offset] = true;
		  			}
		  		}

		  		/* // nms
		  		for(int j = i + 1; j < areas; j++) {
		  			if(skips[j]) {
		  				continue;
		  			}
		  			x2 = tops[j].second % width;
		  	  	y2 = tops[j].second / width;

		  	  	if(x2 >= bx1 && x2 <= bx2 && y2 >= by1 && y2 <= by2) {
		  	  		skips[j] = true;
		  	  	}
		  		}*/
			  }	// end nms
			}	// end finding top k
			CHECK_EQ(this->top_k_, k);	/* not perfect */
		}	// end c
	}	// end n

	// 
	CHECK_EQ(crop_idx, crop_beg_.count());
	CHECK_EQ(idx1,     bottom[1]->count());
}

template <typename Dtype>
void CropPatchFromMaxFeatureFromTopKPositionsLayer<Dtype>::Forward_cpu(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
	for (int i = 0; i < top.size(); ++i) {
		caffe::caffe_set(top[i]->count(), Dtype(0), top[i]->mutable_cpu_data());
	}

	// reset zero ???
	caffe::caffe_set(crop_beg_.count(), 0, crop_beg_.mutable_cpu_data());
	if(this->top_k_ == 1) {
		DeriveCropBeg(bottom, top);
	} else {
		DeriveCropBegFromTopK(bottom, top);
	}

	const int* is_match_channel = is_match_channel_.cpu_data();
	for (int i = 0; i < top.size(); ++i) {
		int crop_idx = 0;
		const int* crop_beg = crop_beg_.cpu_data();

		for (int n = 0; n < bottom[1]->num(); ++n) {
			for(int c = 0; c < bottom[1]->channels(); ++c) {
				// crop top k for each joint/part
				for (int k = 0; k < this->top_k_; k++, crop_idx += 2) {
					// 算出上和左需要填补空格的长度
					const int top_blank = crop_beg[crop_idx] < 0 ? 
							-crop_beg[crop_idx] : 0;	/* y */
					const int left_blank = crop_beg[crop_idx + 1] < 0 ? 
							-crop_beg[crop_idx + 1] : 0;	/* x */

					// 算出实际上的crop的原图的左上角
					// const int actual_crop_y_beg = crop_beg[crop_idx] + top_blank;
					const int actual_crop_x_beg = crop_beg[crop_idx + 1] + left_blank;
					// 算出实际上的crop的原图的右下角
					// const int actual_crop_y_end = MIN(bottom[0]->height(), crop_beg[crop_idx] + crop_h_);
					const int actual_crop_x_end = MIN(bottom[0]->width(), 
							crop_beg[crop_idx + 1] + crop_w_);
					// 算出实际上的crop的宽度
					const int actual_crop_w = actual_crop_x_end - actual_crop_x_beg;

					if (actual_crop_w <= 0) {
						continue;
					}

					// 截取patch
			    for (int bottom_c = 0; bottom_c < bottom[i + 2]->channels(); ++bottom_c) {
			    	if (is_match_channel[i] == 1) {
			    		if (bottom_c != c) continue;
	 		    	}
			    	
	 		    	const int top_c = (is_match_channel[i] == 1) ?
	 		    			(bottom_c * this->top_k_ + k) :
	 		    			(c * this->top_k_ * bottom[i + 2]->channels() + k * bottom[i + 2]->channels() + bottom_c);

	 		    	// copy
			      for (int top_h = top_blank, crop_y = crop_beg[crop_idx] + top_blank;
			      		top_h < top[i]->height() && crop_y < bottom[i + 2]->height(); 
			      		++top_h, ++crop_y) 
			      {
			      	caffe_copy(actual_crop_w,
			      			bottom[i + 2]->cpu_data() + bottom[i + 2]->offset(n, bottom_c, 
			      				crop_y, actual_crop_x_beg),
									top[i]->mutable_cpu_data() + top[i]->offset(n, top_c, 
										top_h, left_blank));
			      } // end copy
				  } // end 截取patch
				}	// end top k
			}	// end c
		}	// end n
	}	// end top size
}

template <typename Dtype>
void CropPatchFromMaxFeatureFromTopKPositionsLayer<Dtype>::Backward_cpu(
		const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, 
    const vector<Blob<Dtype>*>& bottom) 
{
	for (int i = 0; i < top.size(); ++i) {
		if (propagate_down[i + 2]) {

			caffe::caffe_set(
				bottom[i + 2]->count(), 
				Dtype(0.), 
				bottom[i + 2]->mutable_cpu_diff()
			);

			int crop_idx = 0;
			const int* crop_beg = crop_beg_.cpu_data();
			const int* is_match_channel = is_match_channel_.cpu_data();
			for (int n = 0; n < bottom[1]->num(); ++n) {
				for (int c = 0; c < bottom[1]->channels(); ++c) {
					// crop top k for each joint/part
					for (int k = 0; k < this->top_k_; k++, crop_idx += 2) {
						const int top_blank = crop_beg[crop_idx] < 0 ? -crop_beg[crop_idx] : 0;
						const int left_blank = crop_beg[crop_idx + 1] < 0 ? -crop_beg[crop_idx + 1] : 0;

						const int actual_crop_x_beg = crop_beg[crop_idx + 1] + left_blank;
						const int actual_crop_x_end = MIN(bottom[0]->width(), crop_beg[crop_idx + 1] + crop_w_);
						const int actual_crop_w = actual_crop_x_end - actual_crop_x_beg;
						if (actual_crop_w <= 0) {
							continue;
						}

						// 截取patch
				    for (int bottom_c = 0; bottom_c < bottom[i + 2]->channels(); ++bottom_c) {
				    	if (is_match_channel[i] == 1) {
				    		if (bottom_c != c) continue;
		 		    	}
				    	
				    	const int top_c = (is_match_channel[i] == 1) ?
	 		    			(bottom_c * this->top_k_ + k) :
	 		    			(c * this->top_k_ * bottom[i + 2]->channels() + k * bottom[i + 2]->channels() + bottom_c);

	 		    		// copy
							for (int top_h = top_blank, crop_y = crop_beg[crop_idx] + top_blank;
									top_h < top[i]->height() && crop_y < bottom[i + 2]->height(); ++top_h, ++crop_y) 
							{
								caffe_axpy(actual_crop_w, Dtype(1),
										top[i]->cpu_diff() + top[i]->offset(n, top_c, top_h, left_blank),
										bottom[i + 2]->mutable_cpu_diff() + bottom[i + 2]->offset(n, bottom_c, crop_y, actual_crop_x_beg));
							}	// end copy
				    }	// end 截取patch
					}	// end top k
				}	// end c
		  }	// end n
		}	// end if
	}	// end top size
}


#ifdef CPU_ONLY
STUB_GPU(CropPatchFromMaxFeatureFromTopKPositionsLayer);
#endif

INSTANTIATE_CLASS(CropPatchFromMaxFeatureFromTopKPositionsLayer);
REGISTER_LAYER_CLASS(CropPatchFromMaxFeatureFromTopKPositions);

}  // namespace caffe