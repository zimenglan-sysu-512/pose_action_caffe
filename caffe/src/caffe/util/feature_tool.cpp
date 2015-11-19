// Copyright 2015 DDK

#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/io/coded_stream.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv/cv.h>

#include <cmath>

#include "caffe/util/feature_tool.hpp"
#include "caffe/util/util_img.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

const int RGB = 3;

std::vector<float> GetOrientationDx() {
	const int num_orientations_hog = 24;
	const float Pi = 3.1415926535897931f;
  const float angleIncrement = Pi / num_orientations_hog;
  std::vector<float> orientation_Dx;
  for (int idx = 0; idx < num_orientations_hog; idx++) {
    orientation_Dx.push_back(cos(idx * angleIncrement));
  }
  return orientation_Dx;
}

std::vector<float> GetOrientationDy() {
	const int num_orientations_hog = 24;
	const float Pi = 3.1415926535897931f;
  const float angleIncrement = Pi / num_orientations_hog;
  std::vector<float> orientation_Dy;
  for (int idx = 0; idx < num_orientations_hog; idx++) {
    orientation_Dy.push_back(sin(idx * angleIncrement));
  }
  return orientation_Dy;
}

// 
float GetDX(const std::vector<float> orientation_Dx, int orientation) { 
  return orientation_Dx[orientation]; 
}

float GetDY(const std::vector<float> orientation_Dy, int orientation) { 
  return orientation_Dy[orientation]; 
}

float GetDX(const std::vector<float> orientation_Dx, 
	const std::vector<float> orientation_Dy,
	int orientation, float relX, float relY) { 
  return relX * orientation_Dx[orientation] - relY * orientation_Dy[orientation]; 
}

float GetDY(const std::vector<float> orientation_Dx,
	const std::vector<float> orientation_Dy, 
	int orientation, float relX, float relY) { 
  return relX * orientation_Dy[orientation] + relY * orientation_Dx[orientation]; 
}

template <typename Dtype>
cv::Mat BlobToGreyImage(
			const Blob<Dtype>* blob, 
			const int n, 
			const int c, 
			const Dtype scale) 
{
	cv::Mat img(blob->height(), blob->width(), CV_8UC1);
	for (int h = 0; h < img.rows; ++h) {
		for (int w = 0; w < img.cols; ++w) {
			Dtype value = blob->data_at(n, c, h, w) * scale;
			img.at<uchar>(h, w) = cv::saturate_cast<uchar>(value);
		}
	}

	return img;
}
template cv::Mat BlobToGreyImage(
		const Blob<float>* blob, 
		const int n, 
		const int c, 
		const float scale);
template cv::Mat BlobToGreyImage(
		const Blob<double>* blob, 
		const int n, 
		const int c, 
		const double scale);

/*
	url:
*/
template <typename Dtype>
void GetHOGFeature(
		const cv::Mat& img, 
		Blob<Dtype>* features, 
		const int n,
		int num_orinetations, 
		int num_half_orinetations, 
		int block_size, 
		float eps) 
{
	// int blockSizeH = (block_size - 1) / 2;
	// float blockSizeHF = (float)(block_size) / 2.0f;
	// int border = block_size + 1;

	std::vector<float> orientation_Dx = GetOrientationDx();
	std::vector<float> orientation_Dy = GetOrientationDy();

	int feature_offset;
	int width = img.cols;
	int height = img.rows;
	int channels = img.channels();
	CHECK_EQ(channels, RGB) << "image must be color instead of gray";

	Dtype* hist = new Dtype[width * height * num_orinetations];
	Dtype* norm = new Dtype[width * height];

	memset(norm, 0x00, sizeof(Dtype) * width * height);
	memset(hist, 0x00, sizeof(Dtype) * width * height 
			* num_orinetations);

	// compute
	for (int x = 1; x < width - 1; x++) {
		for (int y = 1; y < height - 1; y++) {

			const cv::Vec3b& pix1 = img.at<cv::Vec3b>(y + 1, x);
			const cv::Vec3b& pix2 = img.at<cv::Vec3b>(y - 1, x);
			const cv::Vec3b& pix3 = img.at<cv::Vec3b>(y, x + 1);
			const cv::Vec3b& pix4 = img.at<cv::Vec3b>(y, x - 1);
			
			// first color channel
			float dy = (float)(pix1[0]-pix2[0])/255.0f;
			float dx = (float)(pix3[0]-pix4[0])/255.0f;
			float v = dx*dx + dy*dy;

			// second color channel
			float dy2 = (float)(pix1[1]-pix2[1])/255.0f;
			float dx2 = (float)(pix3[1]-pix4[1])/255.0f;
			float v2 = dx2*dx2 + dy2*dy2;

			// third color channel
			float dy3 = (float)(pix1[2]-pix2[2])/255.0f;
			float dx3 = (float)(pix3[2]-pix4[2])/255.0f;
			float v3 = dx3*dx3 + dy3*dy3;
			
			// pick channel with strongest gradient
			if (v2 > v) {
				v = v2;
				dx = dx2;
				dy = dy2;
			} 
			if (v3 > v) {
				v = v3;
				dx = dx3;
				dy = dy3;
			}

			// snap to orientation
			float best_dot = 0;
			int best_o = 0;
			for (int o = 0; o < num_half_orinetations; o++) {
				float dot = GetDY(orientation_Dy, o) * dx - GetDX(orientation_Dx, o) * dy;
				
				if (dot > best_dot) {
					best_dot = dot;
					best_o = o;
				} else if (-dot > best_dot) {
					best_dot = -dot;
					best_o = o+num_half_orinetations;
				}
			}
			
			v = std::sqrt(v);

			for (int dx = 0; dx < block_size; dx++) {
				for (int dy = 0; dy < block_size; dy++) {
					int xx = x - dx;
					int yy = y - dy;


					Dtype vx0 = Dtype(dx) / Dtype(block_size); 
					// + (Dtype(1) / ((Dtype)block_size * 2));
					Dtype vy0 = Dtype(dy) / Dtype(block_size); 
					// + (Dtype(1) / ((Dtype)block_size * 2));
					Dtype vx1 = 1 - vx0;
					Dtype vy1 = 1 - vy0;

					int x0 = xx;
					int y0 = yy;
					int x1 = x0 + block_size;
					int y1 = y0 + block_size;

					if (x0 >= 0    && y0 >= 0     && x0 < width  && y0 < height) {
						hist[best_o * width * height + y0 * width + x0] += vx1 * vy1 * v;
					}
					if (x1 < width && y0 >= 0     && y0 < height && x1 >= 0 ) {
						hist[best_o * width * height + y0 * width + x1] += vx0 * vy1 * v;
					}
					if (x0 >= 0    && y1 < height && x0 < width  && y0 >= 0) {
						hist[best_o * width * height + y1 * width + x0] += vx1 * vy0 * v;	
					}
					if (x1 < width && y1 < height && x1 >= 0     && y1 >= 0 ) {
						hist[best_o * width * height + y1 * width + x1] +=  vx0 * vy0 *v;
					}
				}
			}			

		}
	}
	
	// compute energy in each block by summing over orientations
	for (int o = 0; o < num_half_orinetations; o++) {
		Dtype *src1 = hist + o * width * height;
		Dtype *src2 = hist + (o+num_half_orinetations) * width * height;

		Dtype *dst = norm;
		Dtype *end = norm + width*height;
		while (dst < end) {
			*(dst++) += (*src1 + *src2) * (*src1 + *src2);
			src1++;
			src2++;
		}
	}

	// compute features
	for (int x = 0; x < width - (block_size * 2); x ++) {
		for (int y = 0; y < height -(block_size * 2); y++) {
			
			Dtype *src, *p, n1, n2, n3, n4;
			p = norm + (y + block_size) * width + (x + block_size);
			n1 = 1.0 / std::sqrt(*p + *(p + block_size) + *(p + (width * block_size)) 
					+ *(p + (width * block_size) + block_size) + eps);
			
			p = norm + y * width + (x + block_size);
			n2 = 1.0 / std::sqrt(*p + *(p + block_size) + *(p + (width * block_size)) 
					+ *(p + (width * block_size) + block_size) + eps);
			
			p = norm + (y + block_size) * width + x;
			n3 = 1.0 / std::sqrt(*p + *(p + block_size) + *(p + (width * block_size)) 
					+ *(p + (width * block_size) + block_size) + eps);
			
			p = norm + y * width + x;
			n4 = 1.0 / std::sqrt(*p + *(p + block_size) + *(p + (width * block_size)) 
					+ *(p + (width * block_size) + block_size) + eps);

			src = hist + (y+block_size) * width + (x + block_size);

			const Dtype threshold_value = Dtype(0.2f);
			const Dtype half_average = Dtype(0.5);

			for(int o = 0; o < num_half_orinetations; o++) {
				Dtype sum = *src + *(src + width * height * num_half_orinetations);
				Dtype h1 = std::min(sum * n1, threshold_value);
				Dtype h2 = std::min(sum * n2, threshold_value);
				Dtype h3 = std::min(sum * n3, threshold_value);
				Dtype h4 = std::min(sum * n4, threshold_value);

				feature_offset = 
						features->offset(n, o, y + block_size, x + block_size);
				features->mutable_cpu_data()[feature_offset] = 
						half_average * (h1 + h2 + h3 + h4);
				// features->set_data_at(
				// 		half_average * (h1 + h2 + h3 + h4),
				// 		n,
				// 		o,
				// 		y + block_size,
				// 		x + block_size
				// );
				src += width * height;
			}
		}
	}

	// delete
	delete[] hist;
	delete[] norm;
}
template void GetHOGFeature<float>(
		const cv::Mat& img, 
		Blob<float>* features, 
		const int n,
		int num_orinetations, 
		int num_half_orinetations, 
		int block_size, 
		float eps);
template void GetHOGFeature<double>(
		const cv::Mat& img, 
		Blob<double>* features, 
		const int n,
		int num_orinetations, 
		int num_half_orinetations, 
		int block_size, 
		float eps);

void GetGradientFeature(
		const cv::Mat& img, 
		cv::Mat& feature, 
		const bool only_gradient /* false: v, dx, dy, true: v */) 
{
	int width = img.cols;
	int height = img.rows;
	const float max_pixel_val = 255.0f;

	int channels = img.channels();
	CHECK_EQ(channels, RGB)
			<< "image must be color instead of gray";
	// 
	feature = cv::Mat(height, width, CV_8UC3, cv::Scalar(0));

	float dx, dy, v;
	float dx2, dy2, v2;
	float dx3, dy3, v3;
	// compute
	for (int x = 1; x < width - 1; x++) {
		for (int y = 1; y < height - 1; y++) {

			const cv::Vec3b& pix1 = img.at<cv::Vec3b>(y + 1, x);
			const cv::Vec3b& pix2 = img.at<cv::Vec3b>(y - 1, x);
			const cv::Vec3b& pix3 = img.at<cv::Vec3b>(y, x + 1);
			const cv::Vec3b& pix4 = img.at<cv::Vec3b>(y, x - 1);
			
			// first color channel
			dy = (float)(pix1[0]-pix2[0]) / max_pixel_val;
			dx = (float)(pix3[0]-pix4[0]) / max_pixel_val;
			v = dx*dx + dy*dy;

			// second color channel
			dy2 = (float)(pix1[1]-pix2[1]) / max_pixel_val;
			dx2 = (float)(pix3[1]-pix4[1]) / max_pixel_val;
			v2 = dx2*dx2 + dy2*dy2;

			// third color channel
			dy3 = (float)(pix1[2]-pix2[2]) / max_pixel_val;
			dx3 = (float)(pix3[2]-pix4[2]) / max_pixel_val;
			v3 = dx3*dx3 + dy3*dy3;
			
			// pick channel with strongest gradient
			if (v2 > v) {
				v = v2;
				dx = dx2;
				dy = dy2;
			} 
			if (v3 > v) {
				v = v3;
				dx = dx3;
				dy = dy3;
			}
			v = std::sqrt(v);
			v *= max_pixel_val;
			dx *= max_pixel_val;
			dy *= max_pixel_val;

			if(!only_gradient) {
				// get gradient
	      feature.at<cv::Vec3b>(y, x)[0] = 
	          static_cast<uchar>((uint8_t)v);
	      // get dx
	      feature.at<cv::Vec3b>(y, x)[1] = 
	          static_cast<uchar>((uint8_t)dx);
	      // get dy
	      feature.at<cv::Vec3b>(y, x)[2] = 
          static_cast<uchar>((uint8_t)dy);	
			} else {
				// get gradient
	      feature.at<cv::Vec3b>(y, x) = 
	          static_cast<uchar>((uint8_t)v);
			}
			// 
		}
	}
}
void GetGradientFeature(
		const cv::Mat& img, 
		Blob<float>* feature,
		const bool only_gradient /* false: v, dx, dy, true: v */);
void GetGradientFeature(
		const cv::Mat& img, 
		Blob<double>* feature,
		const bool only_gradient /* false: v, dx, dy, true: v */);

/* datum: dose not crop */
template<typename Dtype>
void GetGradientFeature(
    const Datum& datum, 
    Dtype* gradient_data, 
    const int gradient_offset,
    const int height,
    const int width,
    const bool only_gradient /* false: v, dx, dy, true: v */) 
{
	// 
  const string& data = datum.data();
  // 
  const int datum_channels = datum.channels();
  const int datum_height = datum.height();
  const int datum_width = datum.width();

  CHECK_EQ(datum_channels, RGB);
  CHECK_EQ(datum_height, height);
  CHECK_EQ(datum_width, width);

  /* start process data */
  Dtype dx, dy, v;
	Dtype dx2, dy2, v2;
	Dtype dx3, dy3, v3;
	Dtype pix1, pix2, pix3, pix4;
  const Dtype max_pixel_val = Dtype(255.);

	int c, out_idx;
  const bool has_uint8 = data.size() > 0;
  CHECK(has_uint8) << "here assume use data not float_data";

  //  datum_element = datum.float_data(out_idx);
  if(!has_uint8) {
  	NOT_IMPLEMENTED;
  }

	for (int h = 1; h < height - 1; ++h) {
	  for (int w = 1; w < width - 1; ++w) {
	  	// 
	  	c = 0;
	    out_idx = (c * height + h + 1) * width + w;
	    pix1 = Dtype(static_cast<Dtype>(
	    		static_cast<uint8_t>(data[out_idx])));
	    // 
	    out_idx = (c * height + h - 1) * width + w;
	    pix2 =Dtype(static_cast<Dtype>(
	    		static_cast<uint8_t>(data[out_idx])));
	    // 
	    out_idx = (c * height + h) * width + (w + 1);
	    pix3 = Dtype(static_cast<Dtype>(
	    		static_cast<uint8_t>(data[out_idx])));
	    // 
	    out_idx = (c * height + h) * width + (w - 1);
	    pix4 = Dtype(static_cast<Dtype>(
	    		static_cast<uint8_t>(data[out_idx])));
	    // 
	    dx = (pix1 - pix2) / max_pixel_val;
	    dy = (pix3 - pix4) / max_pixel_val;
	    v = dx * dx + dy * dy;

	    // 
	    c = 1;
	    out_idx = (c * height + h + 1) * width + w;
	    pix1 = Dtype(static_cast<Dtype>(
	    		static_cast<uint8_t>(data[out_idx])));
	    // 
	    out_idx = (c * height + h - 1) * width + w;
	    pix2 =Dtype(static_cast<Dtype>(
	    		static_cast<uint8_t>(data[out_idx])));
	    // 
	    out_idx = (c * height + h) * width + (w + 1);
	    pix3 = Dtype(static_cast<Dtype>(
	    		static_cast<uint8_t>(data[out_idx])));
	    // 
	    out_idx = (c * height + h) * width + (w - 1);
	    pix4 = Dtype(static_cast<Dtype>(
	    		static_cast<uint8_t>(data[out_idx])));
	    // 
	    dx2 = (pix1 - pix2) / max_pixel_val;
	    dy2 = (pix3 - pix4) / max_pixel_val;
	    v2 = dx2 * dx2 + dy2 * dy2;

	    // 
	    c = 2;
	    out_idx = (c * height + h + 1) * width + w;
	    pix1 = Dtype(static_cast<Dtype>(
	    		static_cast<uint8_t>(data[out_idx])));
	    // 
	    out_idx = (c * height + h - 1) * width + w;
	    pix2 =Dtype(static_cast<Dtype>(
	    		static_cast<uint8_t>(data[out_idx])));
	    // 
	    out_idx = (c * height + h) * width + (w + 1);
	    pix3 = Dtype(static_cast<Dtype>(
	    		static_cast<uint8_t>(data[out_idx])));
	    // 
	    out_idx = (c * height + h) * width + (w - 1);
	    pix4 = Dtype(static_cast<Dtype>(
	    		static_cast<uint8_t>(data[out_idx])));
	    // 
	    dx3 = (pix1 - pix2) / max_pixel_val;
	    dy3 = (pix3 - pix4) / max_pixel_val;
	    v3 = dx3 * dx3 + dy3 * dy3;

	    if(v2 > v) {
	    	v = v2;
	    	dx = dx2;
	    	dy = dy2;
	    }

	    if(v3 > v) {
	    	v = v3;
	    	dx = dy3;
	    	dy = dy3;
	    }

	    v = std::sqrt(v);
			v *= max_pixel_val;
			dx *= max_pixel_val;
			dy *= max_pixel_val;

			if(!only_gradient) {
				// get gradient
				c = 0;
	      out_idx = (c * height + h) * width + w;
	      gradient_data[gradient_offset + out_idx] = v;
	      // get dx
	      c = 1;
	      out_idx = (c * height + h) * width + w;
	      gradient_data[gradient_offset + out_idx] = dx;
	      // get dy
	      c = 2;
	      out_idx = (c * height + h) * width + w;
	      gradient_data[gradient_offset + out_idx] = dy;
			} else {
				// get gradient
				c = 0;
	      out_idx = (c * height + h) * width + w;
	      gradient_data[gradient_offset + out_idx] = v;
			}
		}	  
	}
}
template void GetGradientFeature<float>(
    const Datum& datum, 
    float* gradient_data, 
    const int gradient_offset,
    const int height,
    const int width,
    const bool only_gradient /* false: v, dx, dy, true: v */);
template void GetGradientFeature<double>(
    const Datum& datum, 
    double* gradient_data, 
    const int gradient_offset,
    const int height,
    const int width,
    const bool only_gradient /* false: v, dx, dy, true: v */);

template<typename Dtype>
void GetGradientFeature(
    const Dtype* image_data, 
    const int image_data_offset,
    Dtype* gradient_data, 
    const int gradient_offset,
    const int channels,
    const int height,
    const int width,
    const bool only_gradient /* false: v, dx, dy, true: v */) 
{
  /* start process data */
  const Dtype max_pixel_val = Dtype(255.);

	int c, out_idx;
	if(channels == 1) {
	  Dtype dx, dy, v;
		Dtype pix1, pix2, pix3, pix4;
		for (int h = 1; h < height - 1; ++h) {
		  for (int w = 1; w < width - 1; ++w) {
		  	// 
		  	c = 0;
		    out_idx = (c * height + h + 1) * width + w;
		    pix1 = image_data[image_data_offset + out_idx];
		    // 
		    out_idx = (c * height + h - 1) * width + w;
		    pix2 = image_data[image_data_offset + out_idx];
		    // 
		    out_idx = (c * height + h) * width + w + 1;
		    pix3 = image_data[image_data_offset + out_idx];
		    // 
		    out_idx = (c * height + h) * width + w - 1;
		    pix4 = image_data[image_data_offset + out_idx];
		    // 
		    dx = (pix1 - pix2) / max_pixel_val;
		    dy = (pix3 - pix4) / max_pixel_val;
		    v = dx * dx + dy * dy;

		    v = std::sqrt(v);
				v *= max_pixel_val;
				dx *= max_pixel_val;
				dy *= max_pixel_val;

				if(!only_gradient) {
					// get gradient
					c = 0;
		      out_idx = (c * height + h) * width + w;
		      gradient_data[gradient_offset + out_idx] = v;
		      // get dx
		      c = 1;
		      out_idx = (c * height + h) * width + w;
		      gradient_data[gradient_offset + out_idx] = dx;
		      // get dy
		      c = 2;
		      out_idx = (c * height + h) * width + w;
		      gradient_data[gradient_offset + out_idx] = dy;
				} else {
					// get gradient
					c = 0;
		      out_idx = (c * height + h) * width + w;
		      gradient_data[gradient_offset + out_idx] = v;
				}
			}	  
		}
	// 
	} else if(channels == 3) {
		Dtype dx, dy, v;
		Dtype dx2, dy2, v2;
		Dtype dx3, dy3, v3;
		Dtype pix1, pix2, pix3, pix4;

		for (int h = 1; h < height - 1; ++h) {
		  for (int w = 1; w < width - 1; ++w) {
		  	// 
		  	c = 0;
		    out_idx = (c * height + h + 1) * width + w; 
		    pix1 = image_data[image_data_offset + out_idx];
		    // 
		    out_idx = (c * height + h - 1) * width + w;
		    pix2 = image_data[image_data_offset + out_idx];
		    // 
		    out_idx = (c * height + h) * width + (w + 1);
		    pix3 = image_data[image_data_offset + out_idx];
		    // 
		    out_idx = (c * height + h) * width + (w - 1);
		    pix4 = image_data[image_data_offset + out_idx];
		    // 
		    dx = (pix1 - pix2) / max_pixel_val;
		    dy = (pix3 - pix4) / max_pixel_val;
		    v = dx * dx + dy * dy;
		   
		    // 
		    c = 1;
		    out_idx = (c * height + h + 1) * width + w; 
		    pix1 = image_data[image_data_offset + out_idx];
		    // 
		    out_idx = (c * height + h - 1) * width + w;
		    pix2 = image_data[image_data_offset + out_idx];
		    // 
		    out_idx = (c * height + h) * width + (w + 1);
		    pix3 = image_data[image_data_offset + out_idx];
		    // 
		    out_idx = (c * height + h) * width + (w - 1);
		    pix4 = image_data[image_data_offset + out_idx];
		    // 
		    dx2 = (pix1 - pix2) / max_pixel_val;
		    dy2 = (pix3 - pix4) / max_pixel_val;
		    v2 = dx2 * dx2 + dy2 * dy2;
		   

		    // 
		    c = 2;
		    out_idx = (c * height + h + 1) * width + w; 
		    pix1 = image_data[image_data_offset + out_idx];
		    // 
		    out_idx = (c * height + h - 1) * width + w;
		    pix2 = image_data[image_data_offset + out_idx];
		    // 
		    out_idx = (c * height + h) * width + (w + 1);
		    pix3 = image_data[image_data_offset + out_idx];
		    // 
		    out_idx = (c * height + h) * width + (w - 1);
		    pix4 = image_data[image_data_offset + out_idx];
		    // 
		    dx3 = (pix1 - pix2) / max_pixel_val;
		    dy3 = (pix3 - pix4) / max_pixel_val;
		    v3 = dx3 * dx3 + dy3 * dy3;


		    if(v2 > v) {
		    	v = v2;
		    	dx = dx2;
		    	dy = dy2;
		    }

		    if(v3 > v) {
		    	v = v3;
		    	dx = dy3;
		    	dy = dy3;
		    }

		    v = std::sqrt(v);
				v *= max_pixel_val;
				dx *= max_pixel_val;
				dy *= max_pixel_val;

				if(!only_gradient) {
					// get gradient
					c = 0;
		      out_idx = (c * height + h) * width + w;
		      gradient_data[gradient_offset + out_idx] = v;
		      // get dx
		      c = 1;
		      out_idx = (c * height + h) * width + w;
		      gradient_data[gradient_offset + out_idx] = dx;
		      // get dy
		      c = 2;
		      out_idx = (c * height + h) * width + w;
		      gradient_data[gradient_offset + out_idx] = dy;
				} else {
					// get gradient
					c = 0;
		      out_idx = (c * height + h) * width + w;
		      gradient_data[gradient_offset + out_idx] = v;
				}
			}	  
		}
	// 
	} else {
		LOG(INFO) << "invalid channels: " << channels;
		NOT_IMPLEMENTED;
	}
}
template void GetGradientFeature<float>(
    const float* image_data, 
    const int image_data_offset,
    float* gradient_data, 
    const int gradient_offset,
    const int channels,
    const int height,
    const int width,
    const bool only_gradient /* false: v, dx, dy, true: v */);
template void GetGradientFeature<double>(
    const double* image_data, 
    const int image_data_offset,
    double* gradient_data, 
    const int gradient_offset,
    const int channels,
    const int height,
    const int width,
    const bool only_gradient /* false: v, dx, dy, true: v */);

} // namespace caffe
