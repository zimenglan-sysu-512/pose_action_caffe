#!/usr/bin/env sh

# command 
# 	 cd ../../../caffe/ && make -j8 && cd - && sh deploy_pt.sh

# ########################################
gpu_id=0

pt_file="/home/ddk/download/pose.test.nature.scene/pt_props_m.txt"

cfg_file="/home/ddk/dongdk/pose-caffe/demo/Pose2/flic/deploy_pt.yml"

deploy_pt="/home/ddk/dongdk/pose-caffe/demo/Pose2/flic/deploy_pt.pt"

caffemodel="/home/ddk/dongdk/asserts/models/Pose2/flic/final/refnet_fusnet_tmask_2b_2/models/flic_iter_8000.caffemodel"

log_file="/home/ddk/dongdk/pose-caffe/demo/Pose2/flic/deploy_pt.log"
rm $log_file

python ../../../tools/test_pose_net.py \
		--gpu_id $gpu_id \
		--pt_file $pt_file \
		--cfg_file $cfg_file \
		--deploy_pt $deploy_pt \
		--caffemodel $caffemodel \
		# 2>&1 | tee -a $log_file
