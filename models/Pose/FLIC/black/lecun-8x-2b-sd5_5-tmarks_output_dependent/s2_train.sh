#!/usr/bin/env sh

ProjectDir="../../../../../"
cd $ProjectDir
export LD_LIBRARY_PATH=../tool/cuddn/v2/lib/:$LD_LIBRARY_PATH

CaffeRootDir="caffe/"
# build/tools/
Tools="build/tools/"
Tools=$CaffeRootDir$Tools
# models/Pose/FLIC/black/lecun-8x-2b-sd5_5-tmarks_output_dependent/
ProtosDir="models/Pose/"
SubProtosDir="FLIC/black/"
ExperName="lecun-8x-2b-sd5_5-tmarks_output_dependent/"
ExperPath=$ProtosDir$SubProtosDir$ExperName
mkdir -p $ExperPath
# models/Pose/FLIC/black/lecun-8x-2b-sd5_5-tmarks_output_dependent/s2_solver.pt
solver_proto="s2_solver.pt"
solver_proto=$ExperPath$solver_proto
echo $solver_proto
# ../asserts/models/Pose/FLIC/black/lecun-8x-2b-sd5_5-tmarks_output_dependent/
ModelsDir="../asserts/"
ModelsDir=$ModelsDir$ExperPath
mkdir -p $ModelsDir
# ../asserts/models/Pose/FLIC/black/lecun-8x-2b-sd5_5-tmarks_output_dependent/s2_models/
model_path="s2_models/"
model_path=$ModelsDir$model_path
mkdir -p $model_path
echo $model_path
# ../asserts/models/Pose/FLIC/black/lecun-8x-2b-sd5_5-tmarks_output_dependent/s2_log/
log_path="s2_log/"
log_path=$ModelsDir$log_path
mkdir -p $log_path
# prefix -- lof file
cur_file_prefix="flic_"
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
sleep_time=1
sleep $sleep_time
# resume model file
if [ ! -n "$1" ] ;then
	re_iter=0
	resume_model_file="flic_iter_"$re_iter".caffemodel"
else
	re_iter=$1
	resume_model_file="flic_iter_"$re_iter".solverstate"
fi
resume_model_file=$model_path$resume_model_file
echo
echo "re_iter:" $re_iter
echo "snapshot path:" $resume_model_file
echo

# run & log command
if [ ! -n "$1" ] ;then
	$caffe_bin train --solver=$solver_joint_proto --weights=$resume_model_file 2>&1 | tee -a $log_filename
else
	$caffe_bin train --solver=$solver_joint_proto --snapshot=$resume_model_file 2>&1 | tee -a $log_filename
fi
echo "Done!"