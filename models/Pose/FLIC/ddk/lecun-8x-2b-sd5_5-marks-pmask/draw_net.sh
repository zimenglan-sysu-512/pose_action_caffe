python_dir="../../../../../caffe/python/"
cd $python_dir

rankdir="TB"
draw_net="draw_net.py"
proto_file="../../models/Pose/FLIC/ddk/lecun-8x-2b-sd5_5-marks-pmask/train_val.prototxt"
image_file="../../models/Pose/FLIC/ddk/lecun-8x-2b-sd5_5-marks-pmask/tb_train_val.jpg"

python $draw_net $proto_file $image_file --rankdir=$rankdir