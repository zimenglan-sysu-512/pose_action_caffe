python_dir="../../../../../caffe/python/"
cd $python_dir

rankdir="TB"
draw_net="draw_net.py"
proto_file="../../models/Pose/FLIC/black/lecun-8x-2b-sd5_5-tmarks_mid_c4_2-input/train_val.prototxt"
image_file="../../models/Pose/FLIC/black/lecun-8x-2b-sd5_5-tmarks_mid_c4_2-input/tb_train_val.jpg"

python $draw_net $proto_file $image_file --rankdir=$rankdir