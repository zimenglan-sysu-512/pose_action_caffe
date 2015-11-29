#ifdef WITH_PYTHON_LAYER
#include "boost/python.hpp"
namespace bp = boost::python;
#endif

#include <glog/logging.h>
#include "caffe/caffe.hpp"
#include "caffe/global_variables.hpp"
#include "boost/algorithm/string.hpp"

#include <map>
#include <queue>
#include <string>
#include <vector>
#include <cstring>
#include <sstream>
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

// video
const int _D_F_Width = 640;
const int _D_F_Height = 480;
int _f_width = _D_F_Width;
int _f_height = _D_F_Height;
// pose 
const int _D_Img_Min_Len = 240;
const int _D_Img_Max_Len = 256;
int _img_min_len = _D_Img_Min_Len;
int _img_max_len = _D_Img_Max_Len;
// torso - half size
const int _D_Torso_Width = 30;
const int _D_Torso_Height = 30;
// fps
int _fps = 30;
// mutex
boost::mutex _mutex; 
// frame pool
std::queue<cv::Mat> _frames;
// video name
std::string _video_name = "demo_pose.avi";
// 
const int _radius = 2;
const int _thickness = 2;
const cv::Scalar _blue = cv::Scalar(216, 16, 216);
const cv::Scalar _red = cv::Scalar(255, 0, 0);
const cv::Scalar _yellow = cv::Scalar(255, 255, 0);
// default or use FLAGS
int _part_num = 14;
int _has_torso = 0;
int _batch_size = 1;
int _iterations = 50;
std::string _input_directory = "";
std::string _input_label_file = "";
std::string _output_directory = "";

DEFINE_string(gpu, "",
    "Optional; run in GPU mode on given device IDs separated by ','."
    "Use '-gpu all' to run on all available GPUs. The effective training "
    "batch size is multiplied by the number of devices.");
DEFINE_string(deployprototxt, "",
    "The model definition protocol buffer text file..");
DEFINE_string(trainedmodel, "",
    "Optional; the pretrained weights to initialize finetuning. "
    "Cannot be set simultaneously with snapshot.");
DEFINE_int32(iterations, 50,
    "The number of iterations to run.");
DEFINE_int32(batchsize, 1,
    "The number of images to be process per iteration.");
DEFINE_int32(hastorso, 0,
    "The number of images to be process per iteration.");
DEFINE_int32(partnum, 14,
    "The number of images to be process per iteration.");
DEFINE_int32(fps, 30,
    "The number of images captu_red by video per second.");
DEFINE_int32(fwidth, 640,
    "The width of frame from video .");
DEFINE_int32(fheight, 480,
    "The height of frame from video .");
DEFINE_int32(imgminlen, 240,
    "The width of frame from video .");
DEFINE_int32(imgmaxlen, 256,
    "The height of frame from video .");
DEFINE_string(inputdirectory, "",
    "Optional; the input of the images.");
DEFINE_string(inputlabelfile, "",
    "Optional; the input of label file specifying the input image and its "
    "corresponding label information, like torso/person bbox");
DEFINE_string(outputdirectory, "",
    "Optional; the output of the results, like files or visualized images.");

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

// caffe commands to call by
//     caffe <command> <args>
//
// To add a command, define a function "int command()" and register it with
// RegisterBrewFunction(action);

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

// Load the weights from the specified caffemodel(s) into the train and
// test nets.
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
  _fps = FLAGS_fps;
  _f_width = FLAGS_fwidth;
  _f_height = FLAGS_fheight;
  _part_num = FLAGS_partnum;
  _has_torso = FLAGS_hastorso;
  _batch_size = FLAGS_batchsize;
  _iterations = FLAGS_iterations;
  _img_min_len = FLAGS_imgminlen;
  _img_max_len = FLAGS_imgmaxlen;
  _input_directory = FLAGS_inputdirectory;
  _input_label_file = FLAGS_inputlabelfile;
  _output_directory = FLAGS_outputdirectory;
}

// save results from pose estimation
int _show_results(){
  cv::VideoWriter s_v_fd;
  cv::Size s = cv::Size(_f_width, _f_height);
  // open
  s_v_fd.open(_video_name,CV_FOURCC('M','J','P','G'), _fps, s);
  // save the results to video
  if (!s_v_fd.isOpened()) {
      LOG(INFO)  << "Could not open the output video for write: " 
          << _video_name ;
      return -1;
  }
  while(true) {
    _mutex.lock();
    if (!_frames.empty()) {
      cv::Mat frame = _frames.front();
      _frames.pop();
      cv::resize(frame, frame, s);
      s_v_fd << frame;
      cv::imshow("pose demo", frame);
    }
    _mutex.unlock();
    boost::this_thread::sleep(boost::posix_time::seconds(0.03));
  }

  return 0;
}

// show result from camera.
int _pose_estimate() {
  // init for preparation
  _init();

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

  // capture images from video
  int imgidx = 0;
  string objidx = "0";
  vector<string> imgidxs;
  vector<string> objidxs;
  vector<string> images_paths;

  int t_x = -1;
  int t_y = -1;
  if(_has_torso) {
    t_x = _f_width / 2;
    t_y = _f_height / 2;
  }
  // Start
  while(true){
    // clear
    imgidxs.clear();
    images_paths.clear();
    bottom_vec.clear();
    // open camera
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
      LOG(INFO) << "No camera found.";
      return -1;
    }
    // receive data from camera
    cv::Mat frame;
    std::vector<cv::Mat> img_vec;
    while(img_vec.size() < _batch_size){
      cap >> frame;
      if (!frame.empty()) {
        cv::resize(frame, frame, cv::Size(_f_width, _f_height));
        img_vec.push_back(frame);
        cv::waitKey(10);
        stringstream ss;
        ss << imgidx;
        imgidxs.push_back(ss.str());
        objidxs.push_back(objidx);
        std::string img_path = _output_directory + ss.str() + ".jpg";
        images_paths.push_back(img_path);
        ++imgidx;
      }
    }
    CHECK_EQ(img_vec.size(), _batch_size) 
        << "_batch_size does not match the size of img_vec...";

    // reshape
    data_blob->Reshape(_batch_size, 3, _f_height, _f_width);
    aux_info_blob->Reshape(_batch_size, 5, 1, 1);
    // set aux info
    float *aux_data = aux_info_blob->mutable_cpu_data();
    for(int i =0; i < _batch_size; ++i) {
      int offset = aux_info_blob->offset(i);
      aux_data[offset + 0] = i;
      aux_data[offset + 1] = _f_width;
      aux_data[offset + 2] = _f_height;
      aux_data[offset + 3] = 1.;
      aux_data[offset + 4] = 0;
    }
    // set torso
    if(_has_torso) {
      torso_info_blob = new Blob<float>();
      torso_info_blob->Reshape(_batch_size, _part_num * 2, 1, 1);
      float *torso_info = torso_info_blob->mutable_cpu_data();
      // use `whole` mode
      for(int i =0; i < _batch_size; ++i) {
        int offset = aux_info_blob->offset(i);
        torso_info[offset + 0] = std::min(1, t_x - _D_Torso_Width);
        torso_info[offset + 1] = std::min(1, t_y - _D_Torso_Height);
        torso_info[offset + 2] = std::min(_f_width  - 2, 
            t_x + _D_Torso_Width);
        torso_info[offset + 3] = std::max(_f_height - 2, 
            t_y + _D_Torso_Height);
      }
    }

    // set global info
    caffe::GlobalVars::set_objidxs(objidxs);
    caffe::GlobalVars::set_imgidxs(imgidxs);
    caffe::GlobalVars::set_images_paths(images_paths);

    // forward data
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
    // get number of coord of all joints/parts
    int channels = coord_blob->channels();
    CHECK_EQ(channels, _part_num * 2)
        << "Does not match the channels: " << channels;  
    // visualize the pose
    const float *coord = coord_blob->cpu_data();
    for(int i = 0; i < _batch_size; ++i) {
      // get offset
      int offset = coord_blob->offset(i);
      // one person
      for(int c = 0; c < channels; c += 2){
        float x = coord[offset + c + 0];        
        float y = coord[offset + c + 1];
        cv::Point p(x, y);
        if(c/2 == 4 || c/2 == 5 ){
          cv::circle(img_vec[i], p, _radius, _red, _thickness);
        }else if (c/2 == 7 || c/2 == 8 ) {
          cv::circle(img_vec[i], p, _radius, _yellow, _thickness);
        } else {
          cv::circle(img_vec[i], p, _radius, _blue, _thickness);
        }
      }
      _mutex.lock();
      _frames.push(img_vec[i]);
      _mutex.unlock();
    }
  } 

  return 0;
}

int camera_pose() {
  // init thread
  boost::thread process_thread(_pose_estimate);
  sleep(2);
  boost::thread show_thread(_show_results);
  // start thread
  process_thread.join();
  show_thread.join();

  return 0;
} 
RegisterBrewFunction(camera_pose);

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
      "  camera_pose    show result timely from camera");
  
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
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/camera_pose");
  }
  // exit cleanly on interrupt
  if (quit_signal) exit(0); 
  return 0;
}