#!/usr/bin/env sh

ProjectDir="../../../../../"
cd $ProjectDir

caffe_dire="caffe/"
# build/tools_dire/
tools_dire="build/tools/"
tools_dire=$caffe_dire$tools_dire

# models/Pose2/kinect2/d302/fusnet_tmask_2b_1/
pt_dire="models/Pose2/"
sub_pt_dire="kinect2/d302/"
exper_name="fusnet_tmask_2b_1/"
exper_path=$pt_dire$sub_pt_dire$exper_name
mkdir -p $exper_path

# models/Pose2/kinect2/d302/fusnet_tmask_2b_1/solver.pt
solver_pt="solver.pt"
solver_pt=$exper_path$solver_pt
echo "\n" $solver_pt "\n"

# ../asserts/models/Pose2/kinect2/d302/fusnet_tmask_2b_1/
model_dire="../asserts/"
model_dire=$model_dire$exper_path
mkdir -p $model_dire

# ../asserts/models/Pose2/kinect2/d302/fusnet_tmask_2b_1/models/
model_path="models/"
model_path=$model_dire$model_path
mkdir -p $model_path
echo $model_path

# ../asserts/models/Pose2/kinect2/d302/fusnet_tmask_2b_1/log/
log_path="log/"
log_path=$model_dire$log_path
mkdir -p $log_path

# prefix -- log file
file_prefix="kinect2_"
log_file=$(date -d "today" +"%Y-%m-%d-%H-%M-%S")
log_file=$log_path$file_prefix$log_file".log"

# execute file
caffe_bin="caffe"
caffe_bin=$tools_dire$caffe_bin
echo
echo "######################################"
echo
echo "Usage: "
echo "  sh run.sh [re_iter]"
echo
echo "######################################"
echo

# resume model file
if [ ! -n "$1" ] ;then
	re_iter=0
	resume_model_file="kinect2_iter_"$re_iter".caffemodel"
else
	re_iter=$1
	resume_model_file="kinect2_iter_"$re_iter".solverstate"
fi
resume_model_file=$model_path$resume_model_file
echo
echo "re_iter:" $re_iter
echo "snapshot path:" $resume_model_file
echo

sleep_time=3
sleep $sleep_time

# run & log command
if [ ! -n "$1" ] ;then
	$caffe_bin train --solver=$solver_pt --weights=$resume_model_file 2>&1 | tee -a $log_file
else
	$caffe_bin train --solver=$solver_pt --snapshot=$resume_model_file 2>&1 | tee -a $log_file
fi
echo "Done!"