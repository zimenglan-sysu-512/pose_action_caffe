#!/usr/bin/env sh

project_dir="../../../../../"
caffe_dir="caffe/"
tools="build/tools/"
tools=$project_dir$caffe_dir$tools

deployprototxt="deploy.prototxt"
echo $solver_proto

asserts="../asserts/models/Pose/FLIC/ldp0/"
exper="lecun-8x-2b-addc4_2-sd5_5-marks/"
model="models/flic_iter_14000.caffemodel"
trainedmodel=$project_dir$asserts$exper$model
echo $trainedmodel

log_path="log/"
log_path=$project_dir$asserts$exper$log_path
echo $log_path
mkdir -p $log_path
log_name="deploy.log"
log_path=$log_path$log_name

gpu=0
shoid=6
hipid=10
shoid2=6
hipid2=10
gwidth=100
gheight=100
minsize=240
maxsize=256
partnum=14
hastorso=1
batchsize=10
indirectory="/home/ddk/dongdk/tools/demo/images/double/"
outdirectory="/home/ddk/dongdk/tools/demo/images/v.double/images/"
tpfile="/home/ddk/dongdk/tools/demo/images/v.double/tp_file.log"
mkdir -p $outdirectory

caffe_bin="static_pose"
caffe_bin=$tools$caffe_bin
echo $caffe_bin

sleep_time=3
sleep $sleep_time

$caffe_bin static_pose \
		--gpu=$gpu \
		--shoid=$shoid \
		--hipid=$hipid \
		--shoid2=$shoid2 \
		--hipid2=$hipid2 \
		--gwidth=$gwidth \
		--gheight=$gheight \
		--minsize=$minsize \
		--maxsize=$maxsize \
		--partnum=$partnum \
		--hastorso=$hastorso \
		--batchsize=$batchsize \
		--tpfile=$tpfile \
		--indirectory=$indirectory \
		--outdirectory=$outdirectory \
		--trainedmodel=$trainedmodel \
		--deployprototxt=$deployprototxt \
		2>&1 | tee -a $log_path

echo ""
echo "Test deploy done!"
echo ""