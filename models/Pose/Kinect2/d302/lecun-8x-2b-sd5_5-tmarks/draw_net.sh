python_dir="../../../../../caffe/python/"
cd $python_dir

rankdir="TB"
draw_net="draw_net.py"
proto_file="../../models/Pose/Kinect2/d302/lecun-8x-2b-sd5_5-tmarks/train_val.prototxt"
image_file="../../models/Pose/Kinect2/d302/lecun-8x-2b-sd5_5-tmarks/tb_train_val.jpg"

python $draw_net $proto_file $image_file --rankdir=$rankdir