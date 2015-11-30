#include <vector>

#include "caffe/pose_estimation_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/util_img.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void ResizeLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
	this->has_resize_param_ = this->layer_param_.has_resize_param();
	this->has_bottom_param_ = bottom.size() == 2;
	CHECK((!this->has_resize_param_ && this->has_bottom_param_) || 
		(this->has_resize_param_ && !this->has_bottom_param_)) 
			<<"ResizeParameter or bottom.size() == 2 can not exist at the same \n"
			<< "one of them must be set...";
}

template <typename Dtype>
void ResizeLayer<Dtype>::Reshape(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
	if(this->has_resize_param_) {
		const ResizeParameter resize_param = this->layer_param_.resize_param();
		bool wh = resize_param.has_width() && resize_param.has_height();
		bool wh_fac = resize_param.has_width_fac() && resize_param.has_height_fac();
		CHECK((wh && !wh_fac) || (!wh && wh_fac)) 
				<< "output height&width or height_fac&&width_fac"
				<< " must be required, but can't not both exist...";
		if(wh) {
			this->out_height_ = resize_param.height();
			this->out_width_ = resize_param.width();		
		} else {
			const float height_fac = resize_param.height_fac();
			const float width_fac = resize_param.width_fac();
			this->out_height_ = int(bottom[0]->height() * height_fac);
			this->out_width_ = int(bottom[0]->width() * width_fac);	
		}
	}
	// Configure the kernel size, padding, stride, and inputs
	if(this->has_bottom_param_) {
		bool is_bottom_info = bottom[1]->count() == 4;
		if(is_bottom_info) {
			CHECK(is_bottom_info) 
					<< "output width and height is required by bottom blob";	
			// (width, height, heat_map_a, heat_map_b), 
			// see the layer produces the bottom[1] blob for more details
			this->out_width_ = bottom[1]->cpu_data()[0];
			this->out_height_ = bottom[1]->cpu_data()[1];
		} else {
			this->out_width_ = bottom[1]->width();
			this->out_height_ = bottom[1]->height();
		}
	}
	CHECK_GT(this->out_width_, 0);
	CHECK_GT(this->out_height_, 0);
	
  for(int idx = 0; idx < 4; idx ++) {
	  this->locs_.push_back(new Blob<Dtype>);
  }

  this->out_num_ = bottom[0]->num();
  this->out_channels_ = bottom[0]->channels();

  top[0]->Reshape(
  		this->out_num_, 
  		this->out_channels_, 
  		this->out_height_, 
  		this->out_width_
  );

  for(int idx = 0; idx < 4; ++idx) {
	  this->locs_[idx]->Reshape(
	  		1,
	  		1,
	  		this->out_height_, 
	  		this->out_width_);
  }

  scale_w_ = out_width_ / static_cast<Dtype>(bottom[0]->width());
  scale_h_ = out_height_ / static_cast<Dtype>(bottom[0]->height());
}

template <typename Dtype>
void ResizeLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
	const Blob<Dtype>* bottom_data = bottom[0];
	Blob<Dtype>* top_data = top[0];

	ResizeBlob_cpu(
			bottom_data, 
			top_data,
			this->locs_[0],
			this->locs_[1],
			this->locs_[2],
			this->locs_[3]
	);
}

template <typename Dtype>
void ResizeLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) 
{
	if(!propagate_down[0]) {
		LOG(INFO) << "does not need backward";
		return;
	}
	
	Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
	Dtype* top_diff = top[0]->mutable_cpu_diff();

	const Dtype* loc1 = this->locs_[0]->cpu_data();
	const Dtype* weight1 = this->locs_[0]->cpu_diff();
	const Dtype* loc2 = this->locs_[1]->cpu_data();
	const Dtype* weight2 = this->locs_[1]->cpu_diff();
	const Dtype* loc3 = this->locs_[2]->cpu_data();
	const Dtype* weight3 = this->locs_[2]->cpu_diff();
	const Dtype* loc4 = this->locs_[3]->cpu_data();
	const Dtype* weight4 = this->locs_[3]->cpu_diff();

	caffe::caffe_set(
			bottom[0]->count(),
			Dtype(0),
			bottom_diff
	);

	for(int n=0; n< this->out_num_; ++n) {
		for(int c = 0; c < this->out_channels_; ++c) {
			int bottom_diff_offset = bottom[0]->offset(n,c);
			int top_diff_offset = top[0]->offset(n,c);

			for (int idx = 0; idx < this->out_height_* this->out_width_; ++idx) {
				bottom_diff[bottom_diff_offset + static_cast<int>(loc1[idx])] 
						+= top_diff[top_diff_offset+idx]*weight1[idx];

				bottom_diff[bottom_diff_offset + static_cast<int>(loc2[idx])] 
						+= top_diff[top_diff_offset+idx]*weight2[idx];

				bottom_diff[bottom_diff_offset + static_cast<int>(loc3[idx])] 
						+= top_diff[top_diff_offset+idx]*weight3[idx];

				bottom_diff[bottom_diff_offset + static_cast<int>(loc4[idx])] 
						+= top_diff[top_diff_offset+idx]*weight4[idx];
			}
		}
	}
}

#ifdef CPU_ONLY
STUB_GPU(ResizeLayer);
#endif

INSTANTIATE_CLASS(ResizeLayer);
REGISTER_LAYER_CLASS(Resize);

}  // namespace caffe