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

log_path="log/deploy.log"
log_path=$project_dir$exper$log_path
echo $log_path
mkdir -p $log_path

gpu=0
fps=30
gwidth=100
gheight=100
fwidth=640
fheight=480
partnum=14
hastorso=1
batchsize=5
imgminlen=240
imgmaxlen=256
inputdirectory="~/dongdk/tools/demo/images/double/"
inputlabelfile="~/dongdk/tools/demo/images/v.double/tp_file.log"
outputdirectory="~/dongdk/tools/demo/images/v.double/images/"

caffe_bin="camera_pose"
caffe_bin=$tools$caffe_bin
echo $caffe_bin

sleep_time=3
sleep $sleep_time

$caffe_bin camera_pose \
		--gpu=$gpu \
		--fps=$fps \
		--fwidth=$fwidth \
		--fheight=$fheight \
		--partnum=$partnum \
		--hastorso=$hastorso \
		--imgminlen=$imgminlen \
		--imgmaxlen=$imgmaxlen \
		--inputdirectory=$inputdirectory \
		--inputlabelfile=$inputlabelfile \
		--outputdirectory=$outputdirectory \
		--deployprototxt=$deployprototxt \
		--trainedmodel=$trainedmodel \
		--gwidth=$gwidth \
		--gheight=$gheight \
		2>&1 | tee -a $log_path

echo ""
echo "Test deploy done!"
echo ""