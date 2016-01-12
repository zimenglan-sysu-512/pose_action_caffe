python_dir="../../../../../caffe/python/"
cd $python_dir

rankdir="TB"
draw_net="draw_net.py"
proto_file="../../models/Pose/Kinect2/d302/lecun-8x-2b-sd5_5-tmarks-p2c/train_val.prototxt"
image_file="../../models/Pose/Kinect2/d302/lecun-8x-2b-sd5_5-tmarks-p2c/tb_train_val.jpg"

python $draw_net $proto_file $image_file --rankdir=$rankdir