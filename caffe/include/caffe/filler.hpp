// Fillers are random number generators that fills a blob using the specified
// algorithm. The expectation is that they are only going to be used during
// initialization time and will not involve any GPUs.

#ifndef CAFFE_FILLER_HPP
#define CAFFE_FILLER_HPP

#include <string>

#include "caffe/proto/caffe.pb.h"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"

using namespace std;

namespace caffe {

const std::string None = "None";

/// @brief Fills a Blob with constant or randomly-generated data.
template <typename Dtype>
class Filler {
 public:
  explicit Filler(const FillerParameter& param, 
    const std::string layername = None) 
      : filler_param_(param), layername_(layername) {}

  virtual ~Filler() {}
  virtual void Fill(Blob<Dtype>* blob) = 0;
  virtual void FillDiff(Blob<Dtype>* blob) = 0;

 protected:
  FillerParameter filler_param_;
  std::string layername_;
};  // class Filler

template <typename Dtype>
class ConstantFiller : public Filler<Dtype> {
 public:
  explicit ConstantFiller(const FillerParameter& param,
    const std::string layername = None)
      : Filler<Dtype>(param, layername) {}

  virtual void Fill(Blob<Dtype>* blob) {
    Dtype* data = blob->mutable_cpu_data();
    const int count = blob->count();
    const Dtype value = this->filler_param_.value();
    CHECK(count);

    // init val
    for(int i = 0; i < count; ++i) {
      data[i] = value;
    }

    if(this->layername_ == None) return;

    LOG(INFO) << "*******************";
    LOG(INFO) << "ConstantFiller: Fill Done!";
    LOG(INFO) << "layername: " << this->layername_;
    LOG(INFO) << "value: " << this->filler_param_.value();
    LOG(INFO); 

    CHECK_EQ(this->filler_param_.sparse(), -1)
         << "Sparsity not supported by this Filler.";   
  }

  virtual void FillDiff(Blob<Dtype>* blob) {
    Dtype* diff = blob->mutable_cpu_diff();
    const int count = blob->count();
    const Dtype value = this->filler_param_.value();
    CHECK(count);

    // init val
    for(int i = 0; i < count; ++i) {
      diff[i] = value;
    }

    if(this->layername_ == None) return;

    LOG(INFO) << "*******************";
    LOG(INFO) << "ConstantFiller: Fill Done!";
    LOG(INFO) << "layername: " << this->layername_;
    LOG(INFO) << "value: " << this->filler_param_.value();
    LOG(INFO); 

    CHECK_EQ(this->filler_param_.sparse(), -1)
         << "Sparsity not supported by this Filler.";   
  }
};

/// @brief Fills a Blob with uniformly distributed values @f$ x\sim U(a, b) @f$.
template <typename Dtype>
class UniformFiller : public Filler<Dtype> {
 public:
  explicit UniformFiller(const FillerParameter& param,
    const std::string layername = None)
      : Filler<Dtype>(param, layername) {}

  virtual void Fill(Blob<Dtype>* blob) {
    CHECK(blob->count());
    caffe_rng_uniform<Dtype>(
        blob->count(), 
        Dtype(this->filler_param_.min()),
        Dtype(this->filler_param_.max()), 
        blob->mutable_cpu_data()
    );
    CHECK_EQ(this->filler_param_.sparse(), -1)
         << "Sparsity not supported by this Filler.";

    if(this->layername_ == None) return;

    LOG(INFO) << "*******************";
    LOG(INFO) << "UniformFiller: Fill Done!";
    LOG(INFO) << "layername: " << this->layername_;
    LOG(INFO) << "min value: " << this->filler_param_.min();
    LOG(INFO) << "max value: " << this->filler_param_.max();
    LOG(INFO); 
  }

  virtual void FillDiff(Blob<Dtype>* blob) {
    CHECK(blob->count());
    caffe_rng_uniform<Dtype>(
        blob->count(), 
        Dtype(this->filler_param_.min()),
        Dtype(this->filler_param_.max()), 
        blob->mutable_cpu_diff()
    );
    CHECK_EQ(this->filler_param_.sparse(), -1)
         << "Sparsity not supported by this Filler.";

    if(this->layername_ == None) return;

    LOG(INFO) << "*******************";
    LOG(INFO) << "UniformFiller: Fill Done!";
    LOG(INFO) << "layername: " << this->layername_;
    LOG(INFO) << "min value: " << this->filler_param_.min();
    LOG(INFO) << "max value: " << this->filler_param_.max();
    LOG(INFO); 
  }
};

/// @brief Fills a Blob with Gaussian-distributed values @f$ x = a @f$.
template <typename Dtype>
class GaussianFiller : public Filler<Dtype> {
 public:
  explicit GaussianFiller(const FillerParameter& param,
    const std::string layername = None)
      : Filler<Dtype>(param, layername) {}

  virtual void Fill(Blob<Dtype>* blob) {
    Dtype* data = blob->mutable_cpu_data();
    CHECK(blob->count());
    caffe_rng_gaussian<Dtype>(blob->count(), Dtype(this->filler_param_.mean()),
        Dtype(this->filler_param_.std()), blob->mutable_cpu_data());
    int sparse = this->filler_param_.sparse();
    CHECK_GE(sparse, -1);
    if (sparse >= 0) {
      // Sparse initialization is implemented for "weight" blobs; i.e. matrices.
      // These have num == channels == 1; width is number of inputs; height is
      // number of outputs.  The 'sparse' variable specifies the mean number
      // of non-zero input weights for a given output.
      CHECK_GE(blob->num_axes(), 1);
      const int num_outputs = blob->shape(0);
      Dtype non_zero_probability = Dtype(sparse) / Dtype(num_outputs);
      rand_vec_.reset(new SyncedMemory(blob->count() * sizeof(int)));
      int* mask = reinterpret_cast<int*>(rand_vec_->mutable_cpu_data());
      caffe_rng_bernoulli(blob->count(), non_zero_probability, mask);
      for (int i = 0; i < blob->count(); ++i) {
        data[i] *= mask[i];
      }
    }

    if(this->layername_ == None) return;

    LOG(INFO) << "*******************";
    LOG(INFO) << "GaussianFiller: Fill Done!";
    LOG(INFO) << "layername: " << this->layername_;
    LOG(INFO) << "mean value: " << this->filler_param_.mean();
    LOG(INFO) << "std value: " << this->filler_param_.std();
    LOG(INFO);
  }

  virtual void FillDiff(Blob<Dtype>* blob) {
    Dtype* diff = blob->mutable_cpu_diff();
    CHECK(blob->count());
    caffe_rng_gaussian<Dtype>(blob->count(), Dtype(this->filler_param_.mean()),
        Dtype(this->filler_param_.std()), blob->mutable_cpu_diff());
    int sparse = this->filler_param_.sparse();
    CHECK_GE(sparse, -1);
    if (sparse >= 0) {
      // Sparse initialization is implemented for "weight" blobs; i.e. matrices.
      // These have num == channels == 1; width is number of inputs; height is
      // number of outputs.  The 'sparse' variable specifies the mean number
      // of non-zero input weights for a given output.
      CHECK_GE(blob->num_axes(), 1);
      const int num_outputs = blob->shape(0);
      Dtype non_zero_probability = Dtype(sparse) / Dtype(num_outputs);
      rand_vec_.reset(new SyncedMemory(blob->count() * sizeof(int)));
      int* mask = reinterpret_cast<int*>(rand_vec_->mutable_cpu_data());
      caffe_rng_bernoulli(blob->count(), non_zero_probability, mask);
      for (int i = 0; i < blob->count(); ++i) {
        diff[i] *= mask[i];
      }
    }

    if(this->layername_ == None) return;

    LOG(INFO) << "*******************";
    LOG(INFO) << "GaussianFiller: Fill Done!";
    LOG(INFO) << "layername: " << this->layername_;
    LOG(INFO) << "mean value: " << this->filler_param_.mean();
    LOG(INFO) << "std value: " << this->filler_param_.std();
    LOG(INFO);
  }

 protected:
  shared_ptr<SyncedMemory> rand_vec_;
};

/** @brief Fills a Blob with values @f$ x \in [0, 1] @f$
 *         such that @f$ \forall i \sum_j x_{ij} = 1 @f$.
 */
template <typename Dtype>
class PositiveUnitballFiller : public Filler<Dtype> {
 public:
  explicit PositiveUnitballFiller(const FillerParameter& param,
    const std::string layername = None)
      : Filler<Dtype>(param, layername) {}

  virtual void Fill(Blob<Dtype>* blob) {
    Dtype* data = blob->mutable_cpu_data();
    DCHECK(blob->count());
    caffe_rng_uniform<Dtype>(
        blob->count(), 
        0, 
        1, 
        blob->mutable_cpu_data()
    );

    // We expect the filler to not be called very frequently, so we will
    // just use a simple implementation
    int dim = blob->count() / blob->num();
    CHECK(dim);
    for (int i = 0; i < blob->num(); ++i) {
      Dtype sum = 0;
      for (int j = 0; j < dim; ++j) {
        sum += data[i * dim + j];
      }
      for (int j = 0; j < dim; ++j) {
        data[i * dim + j] /= sum;
      }
    }

    CHECK_EQ(this->filler_param_.sparse(), -1)
         << "Sparsity not supported by this Filler.";

    if(this->layername_ == None) return;

    LOG(INFO) << "*******************";
    LOG(INFO) << "PositiveUnitballFiller: Fill Done!";
    LOG(INFO) << "layername: " << this->layername_;
    LOG(INFO) << "min value: " << 0;
    LOG(INFO) << "max value: " << 1;
    LOG(INFO) << "need normalization between [0, channels * height * width]";
    LOG(INFO) << "((n, c, h, w) = (n, c, h, w) / sum_c'h'w'(n, c', h', w')";
    LOG(INFO);
  }

  virtual void FillDiff(Blob<Dtype>* blob) {
  Dtype* diff = blob->mutable_cpu_diff();
  DCHECK(blob->count());
  caffe_rng_uniform<Dtype>(
      blob->count(), 
      0, 
      1, 
      blob->mutable_cpu_diff()
  );

  // We expect the filler to not be called very frequently, so we will
  // just use a simple implementation
  int dim = blob->count() / blob->num();
  CHECK(dim);
  for (int i = 0; i < blob->num(); ++i) {
    Dtype sum = 0;
    for (int j = 0; j < dim; ++j) {
      sum += diff[i * dim + j];
    }
    for (int j = 0; j < dim; ++j) {
      diff[i * dim + j] /= sum;
    }
  }

  CHECK_EQ(this->filler_param_.sparse(), -1)
       << "Sparsity not supported by this Filler.";

  if(this->layername_ == None) return;

  LOG(INFO) << "*******************";
  LOG(INFO) << "PositiveUnitballFiller: Fill Done!";
  LOG(INFO) << "layername: " << this->layername_;
  LOG(INFO) << "min value: " << 0;
  LOG(INFO) << "max value: " << 1;
  LOG(INFO) << "need normalization between [0, channels * height * width]";
  LOG(INFO) << "((n, c, h, w) = (n, c, h, w) / sum_c'h'w'(n, c', h', w')";
  LOG(INFO);
}
};

/**
 * @brief Fills a Blob with values @f$ x \sim U(-a, +a) @f$ where @f$ a @f$
 *        is set inversely proportional to the number of incoming nodes.
 *
 * A Filler based on the paper [Bengio and Glorot 2010]: Understanding
 * the difficulty of training deep feedforward neuralnetworks, but does not
 * use the fan_out value.
 *
 * It fills the incoming matrix by randomly sampling uniform data from
 * [-scale, scale] where scale = sqrt(3 / fan_in) where fan_in is the number
 * of input nodes. You should make sure the input blob has shape (num, a, b, c)
 * where a * b * c = fan_in.
 *
 * TODO(dox): make notation in above comment consistent with rest & use LaTeX.
 */
template <typename Dtype>
class XavierFiller : public Filler<Dtype> {
 public:
  explicit XavierFiller(const FillerParameter& param,
    const std::string layername = None)
      : Filler<Dtype>(param, layername) {}

  virtual void Fill(Blob<Dtype>* blob) {
    CHECK(blob->count());
    int fan_in = blob->count() / blob->num();
    Dtype scale = sqrt(Dtype(3) / fan_in);

    caffe_rng_uniform<Dtype>(blob->count(), -scale, scale,
        blob->mutable_cpu_data());

    CHECK_EQ(this->filler_param_.sparse(), -1)
         << "Sparsity not supported by this Filler.";

    if(this->layername_ == None) return;
    
    LOG(INFO) << "*******************";
    LOG(INFO) << "XavierFiller: Fill Done!";
    LOG(INFO) << "layername: " << this->layername_;
    LOG(INFO) << "-scale: " << -scale;
    LOG(INFO) << "scale: " << scale;
  }

  virtual void FillDiff(Blob<Dtype>* blob) {
  CHECK(blob->count());
  int fan_in = blob->count() / blob->num();
  Dtype scale = sqrt(Dtype(3) / fan_in);

  caffe_rng_uniform<Dtype>(blob->count(), -scale, scale,
      blob->mutable_cpu_diff());

  CHECK_EQ(this->filler_param_.sparse(), -1)
       << "Sparsity not supported by this Filler.";

  if(this->layername_ == None) return;
  
  LOG(INFO) << "*******************";
  LOG(INFO) << "XavierFiller: Fill Done!";
  LOG(INFO) << "layername: " << this->layername_;
  LOG(INFO) << "-scale: " << -scale;
  LOG(INFO) << "scale: " << scale;
}
};

template <typename Dtype>
class TestLocalFiller : public Filler<Dtype> {
 public:
  explicit TestLocalFiller(const FillerParameter& param,
    const std::string layername = None)
      : Filler<Dtype>(param, layername) {}

  virtual void Fill(Blob<Dtype>* blob) {
    LOG(INFO) << "Doing mutable cpu";
    LOG(INFO) << "blobs" << blob;
    LOG(INFO);
    LOG(INFO) << "**************************";
    LOG(INFO);

    // get data
    Dtype* data = blob->mutable_cpu_data();
    
    LOG(INFO) << "Done Doing mutable cpu";
    
    // CHECK_EQ(blob->channels(), 1);

    for(int n = 0; n < blob->num(); n++) {
      for(int c = 0; c < blob->channels(); c++) {
        for(int j = 0; j < blob->height(); j++) {
          for(int i = 0; i < blob->width(); i++) {
            // set value
            *(data+blob->offset(n, c, j, i)) = i;
          }
        }
      }
    }
  }

  virtual void FillDiff(Blob<Dtype>* blob) {
  LOG(INFO) << "Doing mutable cpu";
  LOG(INFO) << "blobs" << blob;
  LOG(INFO);
  LOG(INFO) << "**************************";
  LOG(INFO);

  // get diff
  Dtype* diff = blob->mutable_cpu_diff();
  
  LOG(INFO) << "Done Doing mutable cpu";
  
  // CHECK_EQ(blob->channels(), 1);

  for(int n = 0; n < blob->num(); n++) {
    for(int c = 0; c < blob->channels(); c++) {
      for(int j = 0; j < blob->height(); j++) {
        for(int i = 0; i < blob->width(); i++) {
          // set value
          *(diff+blob->offset(n, c, j, i)) = i;
        }
      }
    }
  }
}
};

/**
 * @brief Get a specific filler from the specification given in FillerParameter.
 *
 * Ideally this would be replaced by a factory pattern, but we will leave it
 * this way for now.
 */
 // A function to get a specific filler from the specification given in
// FillerParameter. Ideally this would be replaced by a factory pattern,
// but we will leave it this way for now.
template <typename Dtype>
Filler<Dtype>* GetFiller(const FillerParameter& param, 
    std::string layername = None) 
{
  const std::string& type = param.type();
  if (type == "constant") {
    return new ConstantFiller<Dtype>(param, layername);
  } else if (type == "gaussian") {
    return new GaussianFiller<Dtype>(param, layername);
  } else if (type == "positive_unitball") {
    return new PositiveUnitballFiller<Dtype>(param, layername);
  } else if (type == "uniform") {
    return new UniformFiller<Dtype>(param, layername);
  } else if (type == "xavier") {
    return new XavierFiller<Dtype>(param, layername);
  // 
  } else if (type == "test_local") {            // for test - local connect
    return new TestLocalFiller<Dtype>(param, layername);
  } else {
    CHECK(false) << "Unknown filler name: " << param.type();
  }
  return (Filler<Dtype>*)(NULL);
}

}  // namespace caffe

#endif  // CAFFE_FILLER_HPP_