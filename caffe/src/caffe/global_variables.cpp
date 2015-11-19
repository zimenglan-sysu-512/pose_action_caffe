#include <glog/logging.h>
#include <cstdio>
#include <ctime>

#include "caffe/common.hpp"
#include "caffe/global_variables.hpp"

namespace caffe {

shared_ptr<GlobalVars> GlobalVars::singleton_;

GlobalVars::GlobalVars() {}

GlobalVars::~GlobalVars() {}

}  // namespace caffe
