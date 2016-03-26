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

int _p_dxy;
int _sho_id;
int _hip_id;
int _sho_id2;
int _hip_id2;
int _g_width;
int _g_height;
int _min_size;
int _max_size;
int _part_num;
double _ratio;
int _has_torso;
int _batch_size;
bool _draw_text;
bool _draw_skel;
bool _disp_info;
std::string _im_ext;
std::string _in_dire;
std::string _pt_file;
std::string _out_dire;
std::string _skel_path;
std::string _data_layer_name;
std::string _aux_info_layer_name;
std::string _gt_coords_layer_name;
std::vector<int> _s_skel_idxs;
std::vector<int> _e_skel_idxs;
const float zero         = 0;
const int _radius        = 2;
const int _thickness     = 2;
const float _fontScale   = 0.5;
const cv::Scalar _color1 = cv::Scalar(229,  16,  48);
const cv::Scalar _color2 = cv::Scalar(14,  219,  64);
const cv::Scalar _color3 = cv::Scalar(0,     0, 249);
const cv::Scalar _color4 = cv::Scalar(136, 232,  36);
const cv::Scalar _color5 = cv::Scalar(216,  26, 118);
const cv::Scalar _color6 = cv::Scalar(15,   12, 210);
const cv::Scalar _color7 = cv::Scalar(55,  155, 119);
const cv::Scalar _color8 = cv::Scalar(255, 205,  25);
const int _fontFace      = cv::FONT_HERSHEY_SIMPLEX;

DEFINE_string(gpu, "", "Optional; run in GPU mode on given device IDs separated by ','."
    "Use '-gpu all' to run on all available GPUs. The effective training "
    "batch size is multiplied by the number of devices.");
DEFINE_string(def, "", "The model definition protocol buffer text file... (deploy prototxt)");
DEFINE_string(caffemodel, "", "The trained model to test... ");
DEFINE_int32(draw_text, 1, "Draw the text or not.");
DEFINE_int32(disp_info, 0, "Display information or not.");
DEFINE_int32(ratio,   0, "ratio of torso width by hand.");
DEFINE_int32(sho_id,  -1, "The index of the shoulder joint (left or right).");
DEFINE_int32(hip_id,  -1, "The index of the hip joint (left or right).");
DEFINE_int32(sho_id2, -1, "The index of the shoulder joint (left or right).");
DEFINE_int32(hip_id2, -1, "The index of the hip joint (left or right).");
DEFINE_int32(batch_size, 1, "The number of images to be process per iteration.");
DEFINE_int32(has_torso,  0, "pose estimation need torso info?.");
DEFINE_int32(part_num,  14, "The number of parts/joints.");
DEFINE_int32(max_size, 256, "The width  of frame/image.");
DEFINE_int32(min_size, 240, "The height of frame/image.");
DEFINE_int32(p_dxy,      0, "expand the person bbox.");
DEFINE_int32(g_width,  100, "The width  of frame/image (for initialization).");
DEFINE_int32(g_height, 100, "The height of frame/image (for initialization).");
DEFINE_string(im_ext, ".jpg",  "image extesion.");
DEFINE_string(in_dire, "",  "The input of the images.");
DEFINE_string(skel_path, "",  "orders of parts to draw a stickmen.");
DEFINE_string(pt_file, "",  "The input of label file specifying the input image and its corresponding label information, like torso/person bbox");
DEFINE_string(out_dire, "", "The output of the results, like files or visualized images.");
DEFINE_string(data_layer_name, "", "input data layer name.");
DEFINE_string(aux_info_layer_name, "", "aux info laye name.");
DEFINE_string(gt_coords_layer_name, "", "gt coords layer name.");

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
  int p_x1, p_y1;  // person bbox
  int p_x2, p_y2;  
  int t_x1, t_y1;  // torso bbox
  int t_x2, t_y2;  
  float scale;     // scale for input
};

struct TP_Info {
  std::string im_name;
  std::string im_path;
  std::vector<BBox> bboxes;
};

struct Ind {
  int bchidx; // index of current batch
  int imgidx; // the index of the image in images
  int objidx; // the index of the objidx in the `imgidx-th` image
};

void get_skeleton_idxs(const std::string source,
                       std::vector<int>& start_skel_idxs, 
                       std::vector<int>& end_skel_idxs) 
{
  std::ifstream filer(source.c_str());
  CHECK(filer);
  LOG(INFO) << "skeleton filepath: " << source;

  int n_skel;
  int s_idx, e_idx;

  filer >> n_skel;
  LOG(INFO);
  LOG(INFO) << "number of skeleton: " << n_skel;

  for(int idx = 0; idx < n_skel; idx++) {
    filer >> s_idx >> e_idx;
    start_skel_idxs.push_back(s_idx);
    end_skel_idxs.push_back(e_idx);
    LOG(INFO) << "idx: " << idx << ", s_idx: " 
              << s_idx << ", e_idx: " << e_idx;
  }
  LOG(INFO);
}

void _init() {
  CHECK_GT(FLAGS_def.size(), 0) 
      << "Need a deploy prototxt to test...";      // check
  CHECK_GT(FLAGS_caffemodel.size(), 0)   
      << "Need trained model to test...";
 
  vector<int> gpus; // set device id and mode
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
  _p_dxy         = FLAGS_p_dxy;       
  _sho_id        = FLAGS_sho_id;       
  _hip_id        = FLAGS_hip_id;
  _sho_id2       = FLAGS_sho_id2;
  _hip_id2       = FLAGS_hip_id2;
  _pt_file       = FLAGS_pt_file;
  _part_num      = FLAGS_part_num;
  _min_size      = FLAGS_min_size;
  _max_size      = FLAGS_max_size;
  _has_torso     = FLAGS_has_torso;
  _batch_size    = FLAGS_batch_size;
  _im_ext        = FLAGS_im_ext;
  _in_dire       = FLAGS_in_dire;
  _out_dire      = FLAGS_out_dire;
  _g_width       = FLAGS_g_width;
  _g_height      = FLAGS_g_height;
  _draw_text     = FLAGS_draw_text != 0;
  _disp_info     = FLAGS_disp_info != 0; 
  _skel_path     = FLAGS_skel_path;

  _data_layer_name      = FLAGS_data_layer_name;
  _aux_info_layer_name  = FLAGS_aux_info_layer_name;
  _gt_coords_layer_name = FLAGS_gt_coords_layer_name;

  if(!_s_skel_idxs.empty()) _s_skel_idxs.clear();
  if(!_e_skel_idxs.empty()) _e_skel_idxs.clear();
  if(_skel_path.length() > 0) {
    get_skeleton_idxs(_skel_path, _s_skel_idxs, _e_skel_idxs);
  }
  _draw_skel = !_s_skel_idxs.empty() && !_e_skel_idxs.empty();
  if(_draw_skel) {
    CHECK_EQ(_s_skel_idxs.size(), _e_skel_idxs.size());
  }

  caffe::GlobalVars::set_g_width( _g_width);
  caffe::GlobalVars::set_g_height(_g_height);
  _ratio = FLAGS_ratio / 100.0;
  CHECK_GE(_ratio, 0.);
  CHECK_LE(_ratio, 1.);

  // display info
  if(_disp_info) {                              
    LOG(INFO) << "ratio: "      << _ratio;
    LOG(INFO) << "sho id: "     << _sho_id;
    LOG(INFO) << "hip id: "     << _hip_id;
    LOG(INFO) << "sho id2: "    << _sho_id2;
    LOG(INFO) << "hip id2: "    << _hip_id2;
    LOG(INFO) << "min size: "   << _min_size;
    LOG(INFO) << "max size: "   << _max_size;
    LOG(INFO) << "has_torso: "  << _has_torso;
    LOG(INFO) << "batch size: " << _batch_size;
  }
}

void _read_tp_info(std::vector<TP_Info>& tp_infos, 
                   const std::string tp_file, 
                   const int n_obj = 9) 
{
  LOG(INFO) << "Opening file " << tp_file;
  std::ifstream filer(tp_file.c_str());
  CHECK(filer);

  // format: (n_obj = 9)
  //  im_path [[ind p_x1 p_y1 p_x2 p_y2 t_x1 t_y1 t_x2 t_y2] ...]
  std::string line;
  while (getline(filer, line)) {
    TP_Info tp_info;
    boost::trim(line);

    std::vector<std::string> info;
    boost::split(info, line, boost::is_any_of(" "));

    int n_info  = info.size();
    int n_info2 = n_info - 1;
    
    CHECK_GE(n_info, 1);
    CHECK_EQ(n_info2 % n_obj, 0) << "error format: " 
                                 << line << "\n";

    tp_info.im_path = info[0];
    boost::trim(tp_info.im_path);
    cv::Mat im  = cv::imread(tp_info.im_path);
    const int h = im.rows;
    const int w = im.cols;

    int n_obj2 = n_info2 / n_obj;
    for(int j = 0; j < n_obj2; j++) {
      BBox bbox;
      bbox.scale = 1.0;
      int j2     = 1 + j * n_obj;
      // bounding boxes for torso and person
      // ignore `ind`
      // person bbox
      bbox.p_x1 = std::atoi(info[j2 + 1].c_str());  
      bbox.p_y1 = std::atoi(info[j2 + 2].c_str());
      bbox.p_x2 = std::atoi(info[j2 + 3].c_str());
      bbox.p_y2 = std::atoi(info[j2 + 4].c_str());
      // torso bbox
      bbox.t_x1 = std::atoi(info[j2 + 5].c_str());  
      bbox.t_y1 = std::atoi(info[j2 + 6].c_str());
      bbox.t_x2 = std::atoi(info[j2 + 7].c_str());
      bbox.t_y2 = std::atoi(info[j2 + 8].c_str());
      // expand person bbox
      bbox.p_x1 -= _p_dxy;
      bbox.p_x1  = max(bbox.p_x1, 1);
      bbox.p_y1 -= _p_dxy;
      bbox.p_y1  = max(bbox.p_y1, 1);
      bbox.p_x2 += _p_dxy;
      bbox.p_x2  = min(bbox.p_x2, w - 2);
      bbox.p_y2 += _p_dxy;
      bbox.p_y2  = min(bbox.p_y2, h - 2);
      tp_info.bboxes.push_back(bbox);
    } // end for

    tp_infos.push_back(tp_info);
  } // end outer while
}

int _pose_estimate() {
  /* pose estimation from images and person&torso detection results */
  _init();  // init for preparation

  std::vector<TP_Info> tp_infos;
  _read_tp_info(tp_infos, _pt_file);  // get labels
  
  // instantiate the caffe net.
  Net<float> caffe_net(FLAGS_def, caffe::TEST);  
  // copy the trained model
  caffe_net.CopyTrainedLayersFrom(FLAGS_caffemodel);      
  
  // bottom input
  vector<Blob<float>* > bottom_vec;                 
  // input image
  Blob<float> *data_blob       = new Blob<float>(); 
  // aux info
  Blob<float> *aux_info_blob   = new Blob<float>(); 
  // set torso info
  Blob<float> *torso_info_blob = NULL;              

  // variables 
  std::vector<Ind>     ids;       
  std::vector<string>  imgidxs;
  std::vector<string>  objidxs;
  std::vector<string>  im_names;
  std::vector<cv::Mat> ims_vec;
  std::vector<string>  im_paths;

  if(_has_torso) {
    torso_info_blob = new Blob<float>();
  }

  LOG(INFO) << "\n\nStart Pose Estimation...\n\n";

  int s = 0;              // index of image in all test dataset
  while(s < tp_infos.size()) {
    ids.clear();          // clear
    ims_vec.clear();
    imgidxs.clear();
    objidxs.clear();
    im_names.clear();
    bottom_vec.clear();
    im_paths.clear();     // clear
    
    int bs = 0;           // actually batch size
    int is = 0;           // instances' num in the batch images
    int max_width  = -1;
    int max_height = -1;
    
    // indices of joints/parts forming a torso mask
    const int s_idx    = _sho_id   * 2;  
    const int h_idx    = _hip_id   * 2;
    const int s_idx2   = _sho_id2  * 2;
    const int h_idx2   = _hip_id2  * 2;
    const int channels = _part_num * 2;

    std::string im_name;
    std::string out_path;
    std::vector<std::string> info;

    // each batch - preprocess data - 1
    while(bs < _batch_size && s < tp_infos.size()){
      // Pointer Reference
      TP_Info& tp_info          = tp_infos[s];    
      std::vector<BBox>& bboxes = tp_info.bboxes;

      LOG(INFO);
      LOG(INFO) << "im_i: " << s 
                << " im_path:" << tp_info.im_path;
                
      cv::Mat im = cv::imread(tp_info.im_path);
      ims_vec.push_back(im);

      boost::split(info, tp_info.im_path, boost::is_any_of("/"));
      im_name = info[info.size() - 1];
      im_names.push_back(im_name);
      info.clear();

      tp_info.im_name = im_name.substr(0,
                            im_name.find_last_of("."));
      
      for(int j = 0; j < bboxes.size(); j++) {
        // person bbox's height and width
        int pw = bboxes[j].p_x2 - bboxes[j].p_x1 + 1;
        int ph = bboxes[j].p_y2 - bboxes[j].p_y1 + 1;
        int min_size = std::min(pw, ph);
        int max_size = std::max(pw, ph);
        float scale = float(std::max(min_size, _min_size)) 
                    / float(std::min(min_size, _min_size));
        if(scale * max_size > _max_size) {
          scale = float(_max_size) / float(max_size);
        }
        
        { /// in general, we set ratio to be 0.
          int tw          = bboxes[j].t_x2 - bboxes[j].t_x1 + 1;
          bboxes[j].t_x1 += int(tw * _ratio);
          // bboxes[j].t_x2 -= int(tw * _ratio);
        }

        bboxes[j].scale = scale; // reset
        max_width       = std::max(max_width,  int(pw * scale));
        max_height      = std::max(max_height, int(ph * scale));
        
        // here just for convinience
        imgidxs.push_back(tp_info.im_name);     
        objidxs.push_back(boost::to_string(j));
        im_paths.push_back(tp_info.im_path);
        
        // <bchidx, imgidx, objidx>
        Ind ind;            
        ind.imgidx = s;  // index of all input images
        ind.objidx = j;  // index of person in one image
        ind.bchidx = bs; // index in present batchize
        ids.push_back(ind); 
        
        is++; // self-increase
      }
  
      s++;  // self-increase
      bs++; // self-increase
    } // end inner while

    CHECK_EQ(is, ids.size());     CHECK_EQ(is, imgidxs.size());
    CHECK_EQ(is, objidxs.size()); CHECK_EQ(is, im_paths.size());
    CHECK_EQ(bs, ims_vec.size());

    const int n_batch_size = is;  
    data_blob->Reshape(n_batch_size, 3, max_height, max_width);
    float* data = data_blob->mutable_cpu_data();
    caffe::caffe_set(data_blob->count(), zero, data); // fullfill

    // each batch - preprocess data - 2
    for(int j = 0; j < n_batch_size; j++) {
      const int imgidx = ids[j].imgidx;
      const int objidx = ids[j].objidx;
      const int bchidx = ids[j].bchidx;
      const BBox& bbox = tp_infos[imgidx].bboxes[objidx];

      vector<float> coords;
      coords.push_back(bbox.p_x1);
      coords.push_back(bbox.p_y1);
      coords.push_back(bbox.p_x2);
      coords.push_back(bbox.p_y2);

      int w = bbox.p_x2 - bbox.p_x1 + 1;
      int h = bbox.p_y2 - bbox.p_y1 + 1;
      w = int(w * bbox.scale);
      h = int(h * bbox.scale);

      cv::Mat im_crop;
      cv::Size S(w, h);
      caffe::CropAndResizePatch(ims_vec[bchidx], im_crop, 
                                coords, S);
      caffe::ImageDataToBlob(data_blob, j, im_crop);
    } // end for

    // fullfill
    aux_info_blob->Reshape(n_batch_size, 5, 1, 1);
    float *aux_data = aux_info_blob->mutable_cpu_data();
    caffe::caffe_set(aux_info_blob->count(), zero, aux_data); 

    // each batch - preprocess data - 3
    for(int j = 0; j < n_batch_size; j++) {
      const int imgidx = ids[j].imgidx;
      const int objidx = ids[j].objidx;
      const BBox& bbox = tp_infos[imgidx].bboxes[objidx];

      // origin width & height and matched scale
      // imgidx, width, height, scale, flippable
      int o = aux_info_blob->offset(j);  
      aux_data[o + 0] = j;
      aux_data[o + 1] = bbox.p_x2 - bbox.p_x1 + 1;
      aux_data[o + 2] = bbox.p_y2 - bbox.p_y1 + 1;
      aux_data[o + 3] = bbox.scale;
      aux_data[o + 4] = 0;
    } // end for

    // set torso -- crop and scale
    if(_has_torso) {
      torso_info_blob->Reshape(n_batch_size, _part_num * 2, 1, 1);
      float *torso_info = torso_info_blob->mutable_cpu_data();
      // fullfill
      caffe::caffe_set(torso_info_blob->count(), zero, torso_info); 

      // each batch - preprocess data - 4
      // don't use `whole` mode -> maybe error
      // see src/caffe/layers/coords_to_bboxes_masks_layer.cpp 
      // for more details
      for(int j = 0; j < n_batch_size; j++) {
        const int imgidx = ids[j].imgidx;
        const int objidx = ids[j].objidx;
        const BBox& bbox = tp_infos[imgidx].bboxes[objidx];

        int o = torso_info_blob->offset(j);
        // substract the left-top cornor of the person bounding box
        int t_x1   = std::max(1, bbox.t_x1 - bbox.p_x1);
        int t_y1   = std::max(1, bbox.t_y1 - bbox.p_y1);
        int t_x2   = std::max(1, bbox.t_x2 - bbox.p_x1);
        int t_y2   = std::max(1, bbox.t_y2 - bbox.p_y1);
        
        torso_info[o + s_idx  + 0] = t_x1 * bbox.scale;
        torso_info[o + s_idx  + 1] = t_y1 * bbox.scale;
        torso_info[o + h_idx  + 0] = t_x2 * bbox.scale;
        torso_info[o + h_idx  + 1] = t_y2 * bbox.scale;
        torso_info[o + s_idx2 + 0] = t_x1 * bbox.scale;
        torso_info[o + s_idx2 + 1] = t_y1 * bbox.scale;
        torso_info[o + h_idx2 + 0] = t_x2 * bbox.scale;
        torso_info[o + h_idx2 + 1] = t_y2 * bbox.scale;
      } // end for
    } // end if
    
    // reshape
    const shared_ptr<Blob<float> > data_in           = 
              caffe_net.blob_by_name(_data_layer_name);
    data_in->ReshapeLike(*data_blob);
    const shared_ptr<Blob<float> > aux_info_in       = 
              caffe_net.blob_by_name(_aux_info_layer_name);
    aux_info_in->ReshapeLike(*aux_info_blob);
    const shared_ptr<Blob<float> > gt_coords_in = 
              caffe_net.blob_by_name(_gt_coords_layer_name);
    gt_coords_in->ReshapeLike(*torso_info_blob);
    
    // set global info
    caffe::GlobalVars::set_objidxs(objidxs);       
    caffe::GlobalVars::set_imgidxs(imgidxs);
    caffe::GlobalVars::set_images_paths(im_paths);
    
    // forward
    float loss;                              
    bottom_vec.push_back(data_blob);
    bottom_vec.push_back(aux_info_blob);
    if(_has_torso) {
      bottom_vec.push_back(torso_info_blob);
    }
    // forward  
    caffe_net.Forward(bottom_vec, &loss);    

    // predicted coordinates
    const vector<vector<Blob<float>*> > &top_vecs = 
              caffe_net.top_vecs(); 
    const int t_id = top_vecs.size() - 1; 
    const Blob<float> *coord_blob = top_vecs[t_id][0];
    const float *coord = coord_blob->cpu_data();
    CHECK_EQ(coord_blob->num(), n_batch_size);                            // predicted coordinates
    CHECK_EQ(channels, coord_blob->channels()) 
            << "Does not match the channels: " << channels;  

    // visualize
    for(int j = 0; j < n_batch_size; j++) {  
      // get offset
      int o = coord_blob->offset(j);         
      const int bchidx = ids[j].bchidx;
      const int imgidx = ids[j].imgidx;
      const int objidx = ids[j].objidx;
      BBox& bbox = tp_infos[imgidx].bboxes[objidx];

      // recover torso bbox
      { /// in general, we set ratio to be 0.
        int tw     = bbox.t_x2 - bbox.t_x1 + 1;
        bbox.t_x1 -= int(tw * _ratio);
        // bbox.t_x2 += int(tw * _ratio);
      }

      // draw joints
      for(int c = 0; c < channels; c += 2) {  
        int x = int(coord[o + c + 0]);
        int y = int(coord[o + c + 1]);
        x += bbox.p_x1;
        y += bbox.p_y1;

        cv::Point p(x, y);
        cv::circle(ims_vec[bchidx], p, _radius, _color6, 
                   _thickness);

        if(_draw_text) {
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

      // draw lines
      if(_draw_skel) { // index starts from zero
        for(int idx = 0; idx < _s_skel_idxs.size(); idx++) {
          const int s_idx  = _s_skel_idxs[idx];
          const int e_idx  = _e_skel_idxs[idx];
          const int s_idx2 = s_idx * 2;
          const int e_idx2 = e_idx * 2;
          // get point
          const int sx = int(coord[o + s_idx2 + 0] + bbox.p_x1);
          const int sy = int(coord[o + s_idx2 + 1] + bbox.p_y1);
          const cv::Point sp(sx, sy);
          const int ex = int(coord[o + e_idx2 + 0] + bbox.p_x1);
          const int ey = int(coord[o + e_idx2 + 1] + bbox.p_y1);
          const cv::Point ep(ex, ey);
          // 
          cv::line(ims_vec[bchidx], sp, ep, _color5, _thickness);
        } // end s_idx
      }

      int pw = bbox.p_x2 - bbox.p_x1 + 1;
      int ph = bbox.p_y2 - bbox.p_y1 + 1;
      pw = min(pw, ims_vec[bchidx].cols - bbox.p_x1 - 2);
      pw = max(1, pw);
      ph = min(pw, ims_vec[bchidx].rows - bbox.p_y1 - 2);
      ph = max(1, ph);
      if(_disp_info) {
        LOG(INFO) << "p_x1: " << bbox.p_x1 << " p_y1: " << bbox.p_y1;                         
        LOG(INFO) << "p_x2: " << bbox.p_x2 << " p_y2: " << bbox.p_y2;                         
        LOG(INFO) << "pw: " << pw << " ph: " << ph;                         
        LOG(INFO) << "pw: " << ims_vec[bchidx].cols << " ph: " 
                  << ims_vec[bchidx].rows;                         
      }
      // top_left.x, top_left.y, width, height
      cv::Rect rect(bbox.p_x1, bbox.p_y1, pw, ph);
      cv::Mat im_crop = ims_vec[bchidx](rect);
      const std::string out_path2 = _out_dire 
                                  + tp_infos[imgidx].im_name + " _"
                                  + boost::to_string(objidx) + _im_ext;
      if(_disp_info) {
        LOG(INFO) << "out_path2:" << out_path2;                          
      }
      cv::imwrite(out_path2, im_crop);

      // draw person bbox
      cv::Point p1(bbox.p_x1, bbox.p_y1);   
      cv::Point p2(bbox.p_x2, bbox.p_y2);
      cv::rectangle(ims_vec[bchidx], p1, p2, 
                    cv::Scalar(145, 33, 216), 2);

      // draw torso bbox
      cv::Point p3(bbox.t_x1, bbox.t_y1);   
      cv::Point p4(bbox.t_x2, bbox.t_y2);
      cv::rectangle(ims_vec[bchidx], p3, p4, 
                    cv::Scalar(36, 213, 115), 2);

    } // end outer for

    // write into images for visualization
    for(int j = 0; j < ims_vec.size(); j++) { 
      out_path = _out_dire + im_names[j];
      cv::imwrite(out_path, ims_vec[j]);
    }
  } // end outer while

  LOG(INFO);
  LOG(INFO) << "\nEnding of Pose Estimation...\n";
  LOG(INFO);
  return 0;
}

int static_pose_v2() {
  _pose_estimate();
  return 0;
} 
RegisterBrewFunction(static_pose_v2);

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
      "  static_pose_v2    show result timely from camera");
  
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
    gflags::ShowUsageWithFlagsRestrict(argv[0], 
        "tools/static_pose_v2");
  }
  // exit cleanly on interrupt
  if (quit_signal) exit(0); 
  return 0;
}