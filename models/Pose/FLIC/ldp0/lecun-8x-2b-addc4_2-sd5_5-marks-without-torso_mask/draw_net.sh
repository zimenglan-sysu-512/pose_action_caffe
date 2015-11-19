python_dir="../../../../../caffe-fast-rcnn/python/"
cd $python_dir

rankdir="TB"
draw_net="draw_net.py"
proto_file="../../models/Pose/FLIC/ldp0/lecun-8x-2b-addc4_2-sd5_5-marks-without-torso_mask/train_val.prototxt"
image_file="../../models/Pose/FLIC/ldp0/lecun-8x-2b-addc4_2-sd5_5-marks-without-torso_mask/tb_train_val.jpg"

python $draw_net $proto_file $image_file --rankdir=$rankdir