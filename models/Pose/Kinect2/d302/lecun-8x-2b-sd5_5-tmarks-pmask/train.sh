#!/usr/bin/env sh
# export LD_LIBRARY_PATH=/home/liangdp/dongdk/cudnn/v3/lib64:$LD_LIBRARY_PATH

ProjectDir="../../../../../"
cd $ProjectDir

RootDir="caffe/"
Tools="build/tools/"
Tools=$RootDir$Tools

# models/Pose/Kinect2/d302/lecun-8x-2b-sd5_5-tmarks-pmask/
ProtosDir="models/Pose/"
SubProtosDir="Kinect2/d302/"
ExperName="lecun-8x-2b-sd5_5-tmarks-pmask/"
ExperPath=$ProtosDir$SubProtosDir$ExperName
mkdir -p $ExperPath

# models/Pose/Kinect2/d302/lecun-8x-2b-sd5_5-tmarks-pmask/solver.prototxt
solver_proto="solver.prototxt"
solver_proto=$ExperPath$solver_proto
echo $solver_proto

# ../asserts/models/Pose/Kinect2/d302/lecun-8x-2b-sd5_5-tmarks-pmask/
ModelsDir="../asserts/"
ModelsDir=$ModelsDir$ExperPath
mkdir -p $ModelsDir

# ../asserts/models/Pose/Kinect2/d302/lecun-8x-2b-sd5_5-tmarks-pmask/models/
model_path="models/"
model_path=$ModelsDir$model_path
mkdir -p $model_path
echo $model_path

# ../asserts/models/Pose/Kinect2/d302/lecun-8x-2b-sd5_5-tmarks-pmask/log/
log_path="log/"
log_path=$ModelsDir$log_path
mkdir -p $log_path

# prefix -- log file
cur_file_prefix="Kinect2_"
cur_log_filename=$(date -d "today" +"%Y-%m-%d-%H-%M-%S")
log_filename=$log_path$cur_file_prefix$cur_log_filename".log"

# execute file
caffe_bin="caffe"
caffe_bin=$Tools$caffe_bin
echo
echo "######################################"
echo
echo "Usage: "
echo "  sh train_val_.sh [re_iter]"
echo
echo "######################################"
echo
sleep_time=3
sleep $sleep_time

# resume model file
if [ ! -n "$1" ] ;then
	re_iter=0
	$caffe_bin train --solver=$solver_proto 2>&1 | tee -a $log_filename
else
	re_iter=$1
	resume_model_file="Kinect2_iter_"$re_iter".solverstate"
	resume_model_file=$model_path$resume_model_file
	echo
	echo "re_iter:" $re_iter
	echo "snapshot path:" $resume_model_file
	echo
	$caffe_bin train --solver=$solver_proto --snapshot=$resume_model_file 2>&1 | tee -a $log_filename
fi

# # resume model file
# if [ ! -n "$1" ] ;then
# 	re_iter=0
# 	mkdir -p $model_path
# 	resume_model_file="Kinect2_iter_"$re_iter".caffemodel"
# else
# 	re_iter=$1
# 	resume_model_file="Kinect2_iter_"$re_iter".solverstate"
# fi
# resume_model_file=$model_path$resume_model_file
# echo
# echo "re_iter:" $re_iter
# echo "snapshot path:" $resume_model_file
# echo

# # run & log command
# if [ ! -n "$1" ] ;then
# 	$caffe_bin train --solver=$solver_proto --weights=$resume_model_file 2>&1 | tee -a $log_filename
# else
# 	$caffe_bin train --solver=$solver_proto --snapshot=$resume_model_file 2>&1 | tee -a $log_filename
# fi

echo ""
echo "Done!"
echo ""