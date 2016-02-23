#ifdef WITH_PYTHON_LAYER
#include "boost/python.hpp"
namespace bp = boost::python;
#endif

#include <glog/logging.h>
#include "caffe/blob.hpp"
#include "caffe/caffe.hpp"
#include "caffe/util/util_img.hpp"
#include "caffe/global_variables.hpp"
#include "boost/algorithm/string.hpp"
#include "caffe/util/math_functions.hpp"

#include <map>
#include <queue>
#include <string>
#include <vector>
#include <cstring>
#include <sstream>
#include <iostream>
#include <boost/thread.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

using caffe::Blob;
using caffe::Caffe;
using caffe::Net;
using caffe::Layer;
using caffe::Timer;
using caffe::string;
using caffe::vector;
using caffe::Solver;
using caffe::shared_ptr;
using std::ostringstream;
using std::stringstream;

int _sho_id;
int _hip_id;
int _sho_id2;
int _hip_id2;
int _g_width;
double _ratio;
int _g_height;
int _min_size;
int _max_size;
int _part_num;
int _has_torso;
int _batch_size;
bool _is_draw_text;
bool _is_disp_info;
std::string _tp_file;
std::string _img_ext;
const float zero = 0;
const int _radius = 2;
const int _thickness = 2;
std::string _in_directory;
std::string _out_directory;
const cv::Scalar _color1 = cv::Scalar(249,  0,    0);
const cv::Scalar _color2 = cv::Scalar(14,  219,  64);
const cv::Scalar _color3 = cv::Scalar(0,    0,  249);
const cv::Scalar _color4 = cv::Scalar(136,  232, 36);
const cv::Scalar _color5 = cv::Scalar(216, 26,   118);
const cv::Scalar _color6 = cv::Scalar(15,  12,  210);
const cv::Scalar _color7 = cv::Scalar(55,  155, 119);
const cv::Scalar _color8 = cv::Scalar(255, 205,  25);
const int _fontFace = cv::FONT_HERSHEY_SIMPLEX;
const float _fontScale = .5;

DEFINE_string(gpu, "",
    "Optional; run in GPU mode on given device IDs separated by ','."
    "Use '-gpu all' to run on all available GPUs. The effective training "
    "batch size is multiplied by the number of devices.");
DEFINE_string(deployprototxt, "",
    "The model definition protocol buffer text file..");
DEFINE_string(trainedmodel, "",
    "Optional; the pretrained weights to initialize finetuning. "
    "Cannot be set simultaneously with snapshot.");
DEFINE_int32(isdrawtext, 1, "Draw the text.");
DEFINE_int32(isdispinfo, 0, "Display information.");
DEFINE_int32(ratio, 0, "ratio of torso width.");
DEFINE_int32(shoid, -1,
    "The index of the shoulder joint (left or right).");
DEFINE_int32(hipid, -1,
    "The index of the hip joint (left or right).");
DEFINE_int32(shoid2, -1,
    "The index of the shoulder joint (left or right).");
DEFINE_int32(hipid2, -1,
    "The index of the hip joint (left or right).");
DEFINE_int32(batchsize, 1,
    "The number of images to be process per iteration.");
DEFINE_int32(hastorso, 0,
    "The number of images to be process per iteration.");
DEFINE_int32(partnum, 14,
    "The number of images to be process per iteration.");
DEFINE_int32(maxsize, 256,
    "The width of frame from video.");
DEFINE_int32(minsize, 240,
    "The height of frame from video.");
DEFINE_int32(gwidth, 100,
    "The width of frame from video.");
DEFINE_int32(gheight, 100,
    "The height of frame from video.");
DEFINE_string(indirectory, "",
    "Optional; the input of the images.");
DEFINE_string(tpfile, "",
    "Optional; the input of label file specifying the input image and its "
    "corresponding label information, like torso/person bbox");
DEFINE_string(outdirectory, "",
    "Optional; the output of the results, like files or visualized images.");
DEFINE_string(imgext, "", "extension of image, e.g. `.jpg`");

// A simple registry for caffe commands.
typedef int (*BrewFunction)();
typedef std::map<caffe::string, BrewFunction> BrewMap;
BrewMap g_brew_map;

// For solving the problem of `Ctrl+C` error
volatile int quit_signal=0;
#ifdef __unix__
#include <signal.h>
extern "C" void quit_signal_handler(int signum) {
 if (quit_signal!=0) exit(0); // just exit already
 quit_signal=1;
}
#endif

#define RegisterBrewFunction(func) \
namespace { \
class __Registerer_##func { \
 public: /* NOLINT */ \
  __Registerer_##func() { \
    g_brew_map[#func] = &func; \
  } \
}; \
__Registerer_##func g_registerer_##func; \
}

static BrewFunction GetBrewFunction(const caffe::string& name) {
  if (g_brew_map.count(name)) {
    return g_brew_map[name];
  } else {
    LOG(ERROR) << "Available caffe actions:";
    for (BrewMap::iterator it = g_brew_map.begin();
         it != g_brew_map.end(); ++it) {
      LOG(ERROR) << "\t" << it->first;
    }
    LOG(FATAL) << "Unknown action: " << name;
    return NULL;  // not reachable, just to suppress old compiler warnings.
  }
}

// Parse GPU ids or use all available devices
static void get_gpus(vector<int>* gpus) {
  if (FLAGS_gpu == "all") {
    int count = 0;
#ifndef CPU_ONLY
    CUDA_CHECK(cudaGetDeviceCount(&count));
#else
    NO_GPU;
#endif
    for (int i = 0; i < count; ++i) {
      gpus->push_back(i);
    }
  } else if (FLAGS_gpu.size()) {
    vector<string> strings;
    boost::split(strings, FLAGS_gpu, boost::is_any_of(","));
    for (int i = 0; i < strings.size(); ++i) {
      gpus->push_back(boost::lexical_cast<int>(strings[i]));
    }
  } else {
    CHECK_EQ(gpus->size(), 0);
  }
}

// Device Query: show diagnostic information for a GPU device.
int device_query() {
  LOG(INFO) << "Querying GPUs" << FLAGS_gpu;
  vector<int> gpus;
  get_gpus(&gpus);
  for (int i = 0; i < gpus.size(); ++i) {
    caffe::Caffe::SetDevice(gpus[i]);
    caffe::Caffe::DeviceQuery();
  }
  
  return 0;
}
RegisterBrewFunction(device_query);

// Load the weights from the specified caffemodel(s) into the train and test nets.
void CopyLayers(caffe::Solver<float>* solver, const std::string& model_list) {
  std::vector<std::string> model_names;
  boost::split(model_names, model_list, boost::is_any_of(",") );
  for (int i = 0; i < model_names.size(); ++i) {
    LOG(INFO) << "Finetuning from " << model_names[i];
    solver->net()->CopyTrainedLayersFrom(model_names[i]);
    for (int j = 0; j < solver->test_nets().size(); ++j) {
      solver->test_nets()[j]->CopyTrainedLayersFrom(model_names[i]);
    }
  }
}

// ####################################################################
// #####              Pose Estimation From Camera                 #####
// ####################################################################

struct BBox {
  int tx1, ty1;  
  int tx2, ty2;  
  int px1, py1;  
  int px2, py2;  
  float scale;
  float tscore;
  float pscore;
};

struct TP_Info {
  std::string im_dir;
  std::string im_name;
  std::string im_path;
  std::vector<BBox> bboxes;
};

struct Ind {
  int bchidx; // index of current batch
  int imgidx; // the index of the image in images
  int objidx; // the index of the objidx in the `imgidx-th` image
};

void _init() {
  // check
  CHECK_GT(FLAGS_deployprototxt.size(), 0) 
      << "Need a deploy prototxt to test...";
  CHECK_GT(FLAGS_trainedmodel.size(), 0) 
      << "Need trained model to test...";

 // set device id and mode
  vector<int> gpus;
  get_gpus(&gpus);
  if (gpus.size() != 0) {
    LOG(INFO) << "Use GPU with device ID " << gpus[0];
    Caffe::SetDevice(gpus[0]);
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(INFO) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);
  }

  // FLAGS args
  _sho_id = FLAGS_shoid;
  _hip_id = FLAGS_hipid;
  _img_ext = FLAGS_imgext;
  _sho_id2 = FLAGS_shoid2;
  _hip_id2 = FLAGS_hipid2;
  _tp_file = FLAGS_tpfile;
  _part_num = FLAGS_partnum;
  _min_size = FLAGS_minsize;
  _max_size = FLAGS_maxsize;
  _has_torso = FLAGS_hastorso;
  _batch_size = FLAGS_batchsize;
  _in_directory = FLAGS_indirectory;
  _out_directory = FLAGS_outdirectory;
  _g_width  = FLAGS_gwidth;
  _g_height  = FLAGS_gheight;
  _is_draw_text = FLAGS_isdrawtext != 0;
  _is_disp_info = FLAGS_isdispinfo != 0;
  caffe::GlobalVars::set_g_width(_g_width);
  caffe::GlobalVars::set_g_height(_g_height);
  _ratio = FLAGS_ratio / 100.0;
  _ratio = max(min(_ratio, 1.0), 0.);

  if(_is_disp_info) {
    LOG(INFO) << "ratio: " << _ratio;
    LOG(INFO) << "sho id: " << _sho_id;
    LOG(INFO) << "hip id: " << _hip_id;
    LOG(INFO) << "sho id2: " << _sho_id2;
    LOG(INFO) << "hip id2: " << _hip_id2;
    LOG(INFO) << "min size: " << _min_size;
    LOG(INFO) << "max size: " << _max_size;
    LOG(INFO) << "has_torso: " << _has_torso;
    LOG(INFO) << "batch size: " << _batch_size;
  }
}

void _read_tp_info(std::vector<TP_Info>& tp_infos, const std::string tp_file,
    const int n = 9 /*the number of labels of the one instance in image*/) 
{
  LOG(INFO) << "Opening file " << tp_file;
  std::ifstream filer(tp_file.c_str());
  CHECK(filer);

  // format:
  //  im_dir im_name ts1 tbbox1 pbbox1 ts2 tbbox2 pbbox2 ...
  // see `~/dongdk/faster-rcnn/tools/demo_torso.py` for more details
  std::string line;
  while (getline(filer, line)) {
    TP_Info tp_info;
    boost::trim(line);
    std::vector<std::string> info;
    boost::split(info, line, boost::is_any_of(" "));
    int l1 = info.size();
    int l2 = l1 - 2;
    CHECK_GE(l1, 2);
    CHECK_EQ(l2 % n, 0) << "error: " 
        << n << " " << l1 << " " << l2
        << "\n" << line;
    tp_info.im_dir = info[0];
    tp_info.im_name = info[1];
    boost::trim(tp_info.im_dir);
    boost::trim(tp_info.im_name);
    tp_info.im_path = tp_info.im_dir + tp_info.im_name + _img_ext;

    int n2 = l2 / n;
    for(int j = 0; j < n2; j++) {
      int j2 = 2 + j * n;
      BBox bbox;
      bbox.tscore = std::atof(info[j2 + 0].c_str());
      bbox.pscore = bbox.tscore; // here use torso's score for person's score
      bbox.scale = 1.0;
      // bounding boxes for torso and person
      bbox.tx1 = std::atoi(info[j2 + 1].c_str());
      bbox.ty1 = std::atoi(info[j2 + 2].c_str());
      bbox.tx2 = std::atoi(info[j2 + 3].c_str());
      bbox.ty2 = std::atoi(info[j2 + 4].c_str());
      bbox.px1 = std::atoi(info[j2 + 5].c_str());
      bbox.py1 = std::atoi(info[j2 + 6].c_str());
      bbox.px2 = std::atoi(info[j2 + 7].c_str());
      bbox.py2 = std::atoi(info[j2 + 8].c_str());
      tp_info.bboxes.push_back(bbox);
    } // end for

    tp_infos.push_back(tp_info);
  } // end outer while
}

int _pose_estimate() {
  // init for preparation
  _init();

  // get labels
  std::vector<TP_Info> tp_infos;
  _read_tp_info(tp_infos, _tp_file);

  // instantiate the caffe net.
  Net<float> caffe_net(FLAGS_deployprototxt, caffe::TEST);

  // copy the trained model
  caffe_net.CopyTrainedLayersFrom(FLAGS_trainedmodel);

  // bottom input
  vector<Blob<float>* > bottom_vec;
  // input image
  Blob<float> *data_blob = new Blob<float>();
  // aux info
  Blob<float> *aux_info_blob = new Blob<float>();
  // set torso info
  Blob<float> *torso_info_blob = NULL;

  std::vector<Ind> ids;
  std::vector<string> imgidxs;
  std::vector<string> objidxs;
  std::vector<string> im_names;
  std::vector<cv::Mat> ims_vec;
  std::vector<string> images_paths;

  if(_has_torso) {
    torso_info_blob = new Blob<float>();
  }

  LOG(INFO);
  LOG(INFO) << "Start...";
  LOG(INFO);

  int s = 0;
  while(s < tp_infos.size()) {
    // clear
    ids.clear();
    ims_vec.clear();
    imgidxs.clear();
    objidxs.clear();
    im_names.clear();
    bottom_vec.clear();
    images_paths.clear();
    
    int bs = 0;
    int is = 0;
    int max_width = -1;
    int max_height = -1;
    while(bs < _batch_size && s < tp_infos.size()){
      // Pointer Reference
      TP_Info& tp_info = tp_infos[s];
      std::vector<BBox>& bboxes = tp_info.bboxes;

      std::string im_path = tp_info.im_path;
      if(_is_disp_info) {
        LOG(INFO) << im_path;
      }
      cv::Mat im = cv::imread(im_path);
      ims_vec.push_back(im);
      im_names.push_back(tp_info.im_name);
      
      for(int j = 0; j < bboxes.size(); j++) {
        int pw = bboxes[j].px2 - bboxes[j].px1 + 1;
        int ph = bboxes[j].py2 - bboxes[j].py1 + 1;
        int min_size = std::min(pw, ph);
        int max_size = std::max(pw, ph);
        float scale = float(std::max(min_size, _min_size)) 
            / float(std::min(min_size, _min_size));
        if(scale * max_size > _max_size) {
          scale = float(_max_size) / float(max_size);
        }
        
        {
          int tw = bboxes[j].tx2 - bboxes[j].tx1 + 1;
          // int th = bboxes[j].ty2 - bboxes[j].ty1 + 1;
          bboxes[j].tx1 += int(tw * _ratio);
          // bboxes[j].tx2 -= int(tw * _ratio);
        }

        // reset
        bboxes[j].scale = scale;  
        max_width = std::max(max_width, int(pw * scale));
        max_height = std::max(max_height, int(ph * scale));
        
        // here just for convinience (with img_ext)
        imgidxs.push_back(tp_info.im_name); 
        objidxs.push_back(boost::to_string(j));
        images_paths.push_back(im_path);
        // <bchidx, imgidx, objidx>
        Ind ind;
        ind.imgidx = s;
        ind.objidx = j;
        ind.bchidx = bs;
        ids.push_back(ind); 
        // self-increase
        is++;
      }
  
      s++;
      bs++;
    } // end inner while

    CHECK_EQ(is, ids.size());
    CHECK_EQ(is, imgidxs.size());
    CHECK_EQ(is, objidxs.size());
    CHECK_EQ(is, images_paths.size());
    CHECK_EQ(bs, ims_vec.size());

    const int n_batch_size = is;
    data_blob->Reshape(n_batch_size, 3, max_height, max_width);
    float* data = data_blob->mutable_cpu_data();
    caffe::caffe_set(data_blob->count(), zero, data);
    for(int j = 0; j < n_batch_size; j++) {
      const int imgidx = ids[j].imgidx;
      const int objidx = ids[j].objidx;
      const int bchidx = ids[j].bchidx;
      const BBox& bbox = tp_infos[imgidx].bboxes[objidx];

      vector<float> coords;
      coords.push_back(bbox.px1);
      coords.push_back(bbox.py1);
      coords.push_back(bbox.px2);
      coords.push_back(bbox.py2);

      int w = bbox.px2 - bbox.px1 + 1;
      int h = bbox.py2 - bbox.py1 + 1;
      w = int(w * bbox.scale);
      h = int(h * bbox.scale);

      cv::Mat im_crop;
      cv::Size S(w, h);
      caffe::CropAndResizePatch(ims_vec[bchidx], im_crop, coords, S);
      caffe::ImageDataToBlob(data_blob, j, im_crop);
    } // end for

    aux_info_blob->Reshape(n_batch_size, 5, 1, 1);
    float *aux_data = aux_info_blob->mutable_cpu_data();
    caffe::caffe_set(aux_info_blob->count(), zero, aux_data);
    for(int j = 0; j < n_batch_size; j++) {
      const int imgidx = ids[j].imgidx;
      const int objidx = ids[j].objidx;
      const BBox& bbox = tp_infos[imgidx].bboxes[objidx];

      // origin width & height
      int o = aux_info_blob->offset(j);
      aux_data[o + 0] = j;
      aux_data[o + 1] = bbox.px2 - bbox.px1 + 1;
      aux_data[o + 2] = bbox.py2 - bbox.py1 + 1;
      aux_data[o + 3] = bbox.scale;
      aux_data[o + 4] = 0;
    } // end for

    // set torso -- crop and scale
    if(_has_torso) {
      torso_info_blob->Reshape(n_batch_size, _part_num * 2, 1, 1);
      float *torso_info = torso_info_blob->mutable_cpu_data();
      caffe::caffe_set(torso_info_blob->count(), zero, torso_info);
      // don't use `whole` mode -> maybe error
      // see src/caffe/layers/coords_to_bboxes_masks_layer.cpp for more details
      for(int j = 0; j < n_batch_size; j++) {
        const int imgidx = ids[j].imgidx;
        const int objidx = ids[j].objidx;
        const BBox& bbox = tp_infos[imgidx].bboxes[objidx];

        int o = torso_info_blob->offset(j);
        // substract the left-top cornor of the person bounding box
        int tx1 = std::max(1, bbox.tx1 - bbox.px1);
        int ty1 = std::max(1, bbox.ty1 - bbox.py1);
        int tx2 = std::max(1, bbox.tx2 - bbox.px1);
        int ty2 = std::max(1, bbox.ty2 - bbox.py1);
        int s_idx = _sho_id * 2;
        int h_idx = _hip_id * 2;
        int s_idx2 = _sho_id2 * 2;
        int h_idx2 = _hip_id2 * 2;
        torso_info[o + s_idx  + 0] = tx1 * bbox.scale;
        torso_info[o + s_idx  + 1] = ty1 * bbox.scale;
        torso_info[o + h_idx  + 0] = tx2 * bbox.scale;
        torso_info[o + h_idx  + 1] = ty2 * bbox.scale;
        torso_info[o + s_idx2 + 0] = tx1 * bbox.scale;
        torso_info[o + s_idx2 + 1] = ty1 * bbox.scale;
        torso_info[o + h_idx2 + 0] = tx2 * bbox.scale;
        torso_info[o + h_idx2 + 1] = ty2 * bbox.scale;
      } // end for
    } // end if
    
    // Reshape
    const shared_ptr<Blob<float> > data_in = 
        caffe_net.blob_by_name("data");
    const shared_ptr<Blob<float> > aux_info_in = 
        caffe_net.blob_by_name("aux_info");
    const shared_ptr<Blob<float> > gt_pose_coords_in = 
        caffe_net.blob_by_name("gt_pose_coords");
    data_in->ReshapeLike(*data_blob);
    aux_info_in->ReshapeLike(*aux_info_blob);
    gt_pose_coords_in->ReshapeLike(*torso_info_blob);

    // Set global info
    caffe::GlobalVars::set_objidxs(objidxs);
    caffe::GlobalVars::set_imgidxs(imgidxs);
    caffe::GlobalVars::set_images_paths(images_paths);

    float loss;
    bottom_vec.push_back(data_blob);
    bottom_vec.push_back(aux_info_blob);
    if(_has_torso) {
      bottom_vec.push_back(torso_info_blob);
    }
    caffe_net.Forward(bottom_vec, &loss);

    // get final result from the last layer
    const vector<vector<Blob<float>*> > &top_vecs = 
        caffe_net.top_vecs();
    const int t_id = top_vecs.size() - 1; 
    const Blob<float> *coord_blob = top_vecs[t_id][0];
    const float *coord = coord_blob->cpu_data();
    CHECK_EQ(coord_blob->num(), n_batch_size);

    // Visualize
    int channels = _part_num * 2;
    CHECK_EQ(channels, coord_blob->channels())
        << "Does not match the channels: " << channels;  
    for(int j = 0; j < n_batch_size; j++) {
      // get offset
      int o = coord_blob->offset(j);
      // 
      const int bchidx = ids[j].bchidx;
      const int imgidx = ids[j].imgidx;
      const int objidx = ids[j].objidx;
      const BBox& bbox = tp_infos[imgidx].bboxes[objidx];

      cv::Point p1(bbox.tx1, bbox.ty1);
      cv::Point p2(bbox.tx2, bbox.ty2);
      cv::rectangle(ims_vec[bchidx], p1, p2, 
          cv::Scalar(2, 33, 245), 1);
      cv::Point p3(bbox.px1, bbox.py1);
      cv::Point p4(bbox.px2, bbox.py2);
      cv::rectangle(ims_vec[bchidx], p3, p4, 
          cv::Scalar(245, 33, 2), 1);
      
      // one person
      for(int c = 0; c < channels; c += 2) {
        // int x = int(coord[o + c + 0] / bbox.scale);
        // int y = int(coord[o + c + 1] / bbox.scale);
        int x = int(coord[o + c + 0]);
        int y = int(coord[o + c + 1]);
        x += bbox.px1;
        y += bbox.py1;

        cv::Point p(x, y);
        cv::circle(ims_vec[bchidx], p, _radius, _color1, _thickness);

        if(_is_draw_text) {
          const int idx = c / 2;
          const std::string text = boost::to_string(idx);
          if(idx % 2) {
            const cv::Point text_point(x - 5, y - 5);
            cv::putText(ims_vec[bchidx], text, text_point, 
                _fontFace, _fontScale, _color2);
          } else {
            const cv::Point text_point(x + 5, y + 5);
            cv::putText(ims_vec[bchidx], text, text_point, 
                _fontFace, _fontScale, _color2);
          }
        } // end if
      } // end inner fori
    } // end outer for

    // Write
    std::string out_path;
    for(int j = 0; j < ims_vec.size(); j++) {
      out_path = _out_directory + im_names[j] + _img_ext;
      cv::imwrite(out_path, ims_vec[j]);
    }
  } // end outer while

  LOG(INFO);
  LOG(INFO) << "End...";
  LOG(INFO)
  ;
  return 0;
}

int static_pose() {
  _pose_estimate();
  return 0;
} 
RegisterBrewFunction(static_pose);

// ####################################################################
// #####                            Main                          #####
// ####################################################################

int main(int argc, char** argv) {
  #ifdef __unix__
   // listen for ctrl-C
   signal(SIGINT,quit_signal_handler); 
  #endif
  // Print output to stderr (while still logging).
  FLAGS_alsologtostderr = 1;
  // Usage message.
  gflags::SetUsageMessage("command line brew\n"
      "usage: caffe <command> <args>\n\n"
      "commands:\n"
      "  static_pose    show result timely from camera");
  
  // Run tool or show usage.
  caffe::GlobalInit(&argc, &argv);
  if (argc == 2) {
#ifdef WITH_PYTHON_LAYER
  try {
#endif
    return GetBrewFunction(caffe::string(argv[1]))();
#ifdef WITH_PYTHON_LAYER
  } catch (bp::error_already_set) {
    PyErr_Print();
    return 1;
  }
#endif
  } else {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/static_pose");
  }
  // exit cleanly on interrupt
  if (quit_signal) exit(0); 
  return 0;
}