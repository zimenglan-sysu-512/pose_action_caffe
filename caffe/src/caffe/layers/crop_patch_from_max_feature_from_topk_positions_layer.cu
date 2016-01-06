// Copyright 2015 ZhuJin Liang


#include <algorithm>
#include <map>
#include <set>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/net.hpp"
#include "caffe/common.hpp"

#include "caffe/zhu_face_layers.hpp"
#include "caffe/util/util_coords.hpp"
#include "caffe/util/util_pre_define.hpp"

namespace caffe {

template <typename Dtype>
__global__ void find_max_kernel(
		const int n,      
		const Dtype* input,
		int* crop_beg,    
		const Dtype* coefs,
		const int step,   
		const int width,
		const int crop_h, 
		const int crop_w) 
{
  CUDA_KERNEL_LOOP(index, n) {
		int max_idx 		 = 0;
  	const int offset = index * step;
		Dtype max_score  = input[offset];

		for (int i = 1; i < step; ++i) {
			if (input[offset + i] > max_score) {
				max_score = input[offset + i];
				max_idx = i;
			}
		}

		crop_beg[index * 2] = std::floor((coefs[0] * (max_idx / width) + coefs[1])
				- crop_h / Dtype(2.));
		crop_beg[index * 2 + 1] = std::floor((coefs[2] * (max_idx % width) + coefs[3])
				- crop_w / Dtype(2.));
  }
}

template <typename Dtype>
__global__ void find_max_kernel_from_topK(
		const int n, 
		const int topK,   
		const int nms_x,  
		const int nms_y,     
		const Dtype* input,
		int* crop_beg,    
		const Dtype* coefs,
		int* skips,
		const int step,   
		const int width,
		const int height,
		const int crop_h, 
		const int crop_w) 
{
  CUDA_KERNEL_LOOP(index, n) {
  	// find top k
		for(int k = 0; k < topK; k++) {
			int max_idx 		 = -1;
			Dtype max_score  = Dtype(-FLT_MAX);
	  	const int offset = index * step;

	  	// finding maximum score and cooresponding coordinates
			for (int i = 0; i < step; ++i) {
				if (skips[offset + i] == 0 && input[offset + i] > max_score) {
					max_score = input[offset + i];
					max_idx = i;
				}
			}	// end finding maximum score and cooresponding coordinates

			int x = max_idx % width;
			int y = max_idx / width;

			int index2 = index * topK + k;
			crop_beg[index2 * 2 + 0] = std::floor((coefs[0] * y + coefs[1])
					- crop_h / Dtype(2.));
			crop_beg[index2 * 2 + 1] = std::floor((coefs[2] * x + coefs[3])
					- crop_w / Dtype(2.));

			// nms
			if(k < topK - 1) {
				int x1 = max(0, 			   x - nms_x);
				int y1 = max(0, 			   y - nms_y);
				int x2 = min(width  - 1, x + nms_x);
				int y2 = min(height - 1, y + nms_y);
				for(int p = x1; p <= x2; p++) {
					for(int q = y1; q <= y2; q++) {
						int nms_offset = q * width + p;
						skips[offset + nms_offset] = 1;
					}
				}
			}	// end nms
		}	// end finding top k		
  }	// end cuda kernel loop
}

template <typename Dtype>
__global__ void crop_patch_forward_kernel(const int nthreads,
		const Dtype* src, Dtype* dst,
		const int* crop_beg, const bool match_channel,
		const int src_channels, const int src_height, const int src_width,
		const int dst_channels, const int dst_height, const int dst_width,
		const int bottom1_channels) {
  CUDA_KERNEL_LOOP(index, nthreads) {

  	// 假设index是dst的下标
  	// 如何从index推出所有的n, c, h, w
  	// 要推出src的，dst的，以及bottom1的
  	// 三者的n都会相同

  	// dst的很好得到：
  	int dst_index = index;
  	const int dst_w = dst_index % dst_width;
  	dst_index /= dst_width;
  	const int dst_h = dst_index % dst_height;
  	dst_index /= dst_height;
  	const int dst_c = dst_index % dst_channels;
  	const int n = dst_index /= dst_channels;

  	// bottom1/dst/src的channel满足如下公式:
  	//  	dst_c = match_channel ? src_c : (bottom1_c * src_channels + src_c);
  	// 				(1) match_channel为false，那么通过求余和除法就可以得到bottom1/src的channel
  	// 				(2) match_channel为true，由上面公式只能得到src的channel，但是注意到一个是，这时候肯定会有
  	// 						bottom1_c = dst_c = src_c的，因为只会截取对应channel

  	// bottom1的
  	const int bottom1_c = match_channel ? dst_c : (dst_c / src_channels);
  	const int bottom1_offset = (n * bottom1_channels + bottom1_c) * 2;

  	// src的
  	const int src_c = match_channel ? dst_c : (dst_c % src_channels);
  	const int src_h = dst_h + crop_beg[bottom1_offset];
  	if (src_h < 0 || src_h >= src_height) continue;
  	const int src_w = dst_w + crop_beg[bottom1_offset + 1];
		if (src_w < 0 || src_w >= src_width) continue;
		const int src_index = (((n * src_channels) + src_c) * src_height + src_h) * src_width + src_w;

		dst[index] = src[src_index];
  }
}

template <typename Dtype>
void CropPatchFromMaxFeatureFromTopKPositionsLayer<Dtype>::Forward_gpu(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

	for (int i = 0; i < top.size(); ++i) {
		caffe::caffe_gpu_set(top[i]->count(), Dtype(0), top[i]->mutable_gpu_data());
	}

	const int step  = bottom[1]->height() * bottom[1]->width();
	const int count = bottom[1]->num()    * bottom[1]->channels();

	find_max_kernel<Dtype><<<CAFFE_GET_BLOCKS(count), 
			CAFFE_CUDA_NUM_THREADS>>>(
	        count, bottom[1]->gpu_data(),
	        crop_beg_.mutable_gpu_data(), coefs_.gpu_data(),
	        step, bottom[1]->width(),
	        crop_h_, crop_w_);
	CUDA_POST_KERNEL_CHECK;

	const int* is_match_channel = is_match_channel_.cpu_data();
	for (int i = 0; i < top.size(); ++i) {

		crop_patch_forward_kernel<Dtype><<<CAFFE_GET_BLOCKS(top[i]->count()), 
				CAFFE_CUDA_NUM_THREADS>>>(
					top[i]->count(),
					bottom[i + 2]->gpu_data(), 
					top[i]->mutable_gpu_data(),
					crop_beg_.gpu_data(), 
					(is_match_channel[i] == 1),
					bottom[i + 2]->channels(), 
					bottom[i + 2]->height(), 
					bottom[i + 2]->width(),
					top[i]->channels(), 
					top[i]->height(), 
					top[i]->width(),
					bottom[1]->channels());
		CUDA_POST_KERNEL_CHECK;
	}
}

template <typename Dtype>
void CropPatchFromMaxFeatureFromTopKPositionsLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	for (int i = 0; i < top.size(); ++i) {
		if (propagate_down[i + 2]) {

			caffe::caffe_gpu_set(bottom[i + 2]->count(), Dtype(0.), bottom[i + 2]->mutable_gpu_diff());

			int crop_idx = 0;
			const int* crop_beg = crop_beg_.cpu_data();
			const int* is_match_channel = is_match_channel_.cpu_data();
			for (int n = 0; n < bottom[1]->num(); ++n) {
				for (int c = 0; c < bottom[1]->channels(); ++c, crop_idx += 2) {

					const int top_blank = crop_beg[crop_idx] < 0 ? -crop_beg[crop_idx] : 0;
					const int left_blank = crop_beg[crop_idx + 1] < 0 ? -crop_beg[crop_idx + 1] : 0;

					const int actual_crop_x_beg = crop_beg[crop_idx + 1] + left_blank;
					const int actual_crop_x_end = MIN(bottom[0]->width(), crop_beg[crop_idx + 1] + crop_w_);
					const int actual_crop_w = actual_crop_x_end - actual_crop_x_beg;
					if (actual_crop_w <= 0) {
						continue;
					}

			    for (int bottom_c = 0; bottom_c < bottom[i + 2]->channels(); ++bottom_c) {
			    	if (is_match_channel[i] == 1) {
			    		if (bottom_c != c) continue;
	 		    	}
			    	const int top_c = (is_match_channel[i] == 1) ? bottom_c : (c * bottom[i + 2]->channels() + bottom_c);

						for (int top_h = top_blank, crop_y = crop_beg[crop_idx] + top_blank;
								top_h < top[i]->height() && crop_y < bottom[i + 2]->height(); ++top_h, ++crop_y) {

							caffe_gpu_axpy(actual_crop_w, Dtype(1),
									top[i]->gpu_diff() + top[i]->offset(n, top_c, top_h, left_blank),
									bottom[i + 2]->mutable_gpu_diff() + bottom[i + 2]->offset(n, bottom_c, crop_y, actual_crop_x_beg));
						}
			    }
				}
		  }
		}
	}
}
INSTANTIATE_LAYER_GPU_FUNCS(CropPatchFromMaxFeatureFromTopKPositionsLayer);


}  // namespace caffe