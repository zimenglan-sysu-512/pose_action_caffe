#!/usr/bin/env sh

# command 
# 	 cd ../../../../../caffe/ && make -j8 && cd - && sh deploy.sh

# ########################################
gpu=0					# set params
p_dxy=10
ratio=20
sho_id=9
hip_id=13
sho_id2=9
hip_id2=13
part_num=19
has_torso=1
draw_text=1
disp_info=1
g_width=100
g_height=100
min_size=256
max_size=320
batch_size=1	# set params
# ########################################

# in_dire=""	# no need so far

out_dire="/home/ddk/malong/dataset/person.torso/demo/vision/mude.images.pose1/"
mkdir -p $out_dire

pt_file="/home/ddk/malong/dataset/person.torso/demo/vision/files/mude.images.per.tor.bbox.txt"

caffemodel="/home/ddk/dongdk/asserts/models/Pose/Kinect2/d302/lecun-8x-2b-sd5_5-tmarks/models/Kinect2_iter_13000.caffemodel"

def="deploy.pt"

log_path="deploy.log"

caffe_bin="/home/ddk/dongdk/pose-caffe/caffe/build/tools/static_pose_v2"

s_time=3
sleep $s_time

$caffe_bin static_pose_v2 \
		--gpu=$gpu \
		--def=$def \
		--p_dxy=$p_dxy \
		--ratio=$ratio \
		--sho_id=$sho_id \
		--hip_id=$hip_id \
		--sho_id2=$sho_id2 \
		--hip_id2=$hip_id2 \
		--g_width=$g_width \
		--pt_file=$pt_file \
		--g_height=$g_height \
		--min_size=$min_size \
		--max_size=$max_size \
		--part_num=$part_num \
		--out_dire=$out_dire \
		--draw_text=$draw_text \
		--has_torso=$has_torso \
		--disp_info=$disp_info \
		--batch_size=$batch_size \
		--caffemodel=$caffemodel \
		2>&1 | tee -a $log_path
