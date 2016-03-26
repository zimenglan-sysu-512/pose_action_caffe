#!/usr/bin/env sh

# command 
# 	 cd ../../../caffe/ && make -j8 && cd - && sh deploy.sh

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

caffe_dire="/home/ddk/dongdk/pose-caffe/caffe/"

im_dire="/home/ddk/dongdk/dataset/demo/pose/mude.images1/"

out_dire="${im_dire}pose.vision/"
mkdir -p $out_dire

pt_file="${im_dire}tp.result/pt.bbox.txt"

skel_path="${caffe_dire}demo/skel_paths/kinect2_19.txt"

model_dire="/home/ddk/dongdk/asserts/models/Pose/Kinect2/d302/"
exper_name="lecun-8x-2b-sd5_5-tmarks/models/"
model_name="Kinect2_iter_13000.caffemodel"
caffemodel="${model_dire}${exper_name}${model_name}"

def="deploy.pt"

log_path="deploy.log"

caffe_bin="${caffe_dire}build/tools/static_pose_v2"

s_time=2
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
		--skel_path $skel_path \
		--draw_text=$draw_text \
		--has_torso=$has_torso \
		--disp_info=$disp_info \
		--batch_size=$batch_size \
		--caffemodel=$caffemodel \
		2>&1 | tee -a $log_path
