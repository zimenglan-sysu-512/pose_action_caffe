python_dir="../../../../../caffe/python/"
cd $python_dir

rankdir="TB"
draw_net="draw_net.py"
proto_file="../../models/Pose/FLIC/lecun-8x-2b-sd5_5-tmarks-output_simple/train_val.pt"
image_file="../../models/Pose/FLIC/lecun-8x-2b-sd5_5-tmarks-output_simple/tb_train_val.jpg"

python $draw_net $proto_file $image_file --rankdir=$rankdir