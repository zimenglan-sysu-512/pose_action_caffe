#!/usr/bin/env sh

ProjectDir="../../../../../"
cd $ProjectDir
export LD_LIBRARY_PATH=../tool/cuddn/v2/lib/:$LD_LIBRARY_PATH

caffe_dire="caffe/"
# build/tools_dire/
tools_dire="build/tools/"
tools_dire=$caffe_dire$tools_dire

# models/Pose2/flic/d302/tmask_2b/
pt_dire="models/Pose2/"
sub_pt_dire="flic/d302/"
exper_name="tmask_2b/"
exper_path=$pt_dire$sub_pt_dire$exper_name
mkdir -p $exper_path

# models/Pose2/flic/d302/tmask_2b/solver.pt
solver_pt="solver.pt"
solver_pt=$exper_path$solver_pt
echo "\n" $solver_pt "\n"

# ../asserts/models/Pose2/flic/d302/tmask_2b/
model_dire="../asserts/"
model_dire=$model_dire$exper_path
mkdir -p $model_dire

# ../asserts/models/Pose2/flic/d302/tmask_2b/models/
model_path="models/"
model_path=$model_dire$model_path
mkdir -p $model_path
echo $model_path

# ../asserts/models/Pose2/flic/d302/tmask_2b/log/
log_path="log/"
log_path=$model_dire$log_path
mkdir -p $log_path

# prefix -- log file
file_prefix="flic_"
log_file=$(date -d "today" +"%Y-%m-%d-%H-%M-%S")
log_file=$log_path$file_prefix$log_file".log"

# execute file
caffe_bin="caffe"
caffe_bin=$tools_dire$caffe_bin
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
	# run & log command
	$caffe_bin train --solver=$solver_pt 2>&1 | tee -a $log_file
else
	re_iter=$1
	resume_model_file="flic_iter_"$re_iter".solverstate"
	resume_model_file=$model_path$resume_model_file
	echo
	echo "re_iter:" $re_iter
	echo "snapshot path:" $resume_model_file
	echo
	# run & log command
	$caffe_bin train --solver=$solver_pt --snapshot=$resume_model_file 2>&1 | tee -a $log_file
fi

echo "Done!"