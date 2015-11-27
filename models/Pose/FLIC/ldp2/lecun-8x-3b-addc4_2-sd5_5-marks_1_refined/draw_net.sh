python_dir="../../../../../caffe/python/"
cd $python_dir

rankdir="TB"
draw_net="draw_net.py"
proto_file="../../models/Pose/FLIC/ldp2/lecun-8x-3b-addc4_2-sd5_5-marks_1_refined/train_val.prototxt"
image_file="../../models/Pose/FLIC/ldp2/lecun-8x-3b-addc4_2-sd5_5-marks_1_refined/tb_train_val.jpg"

python $draw_net $proto_file $image_file --rankdir=$rankdir