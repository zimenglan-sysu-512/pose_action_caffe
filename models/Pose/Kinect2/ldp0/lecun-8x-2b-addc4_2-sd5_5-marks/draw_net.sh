python_dir="../../../../../caffe/python/"
cd $python_dir

rankdir="TB"
draw_net="draw_net.py"
proto_file="../../models/Pose/Kinect2/ldp0/lecun-8x-2b-addc4_2-sd5_5-marks/train_val.prototxt"
image_file="../../models/Pose/Kinect2/ldp0/lecun-8x-2b-addc4_2-sd5_5-marks/tb_train_val.jpg"

python $draw_net $proto_file $image_file --rankdir=$rankdir