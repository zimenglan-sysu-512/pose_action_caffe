python_dir="../../../../../caffe/python/"
cd $python_dir

rankdir="TB"
draw_net="draw_net.py"
proto_file="../../models/Pose2/kinect2_motion/d302/refnet_fusnet_tmask_2b_2/train.pt"
image_file="../../models/Pose2/kinect2_motion/d302/refnet_fusnet_tmask_2b_2/net.jpg"

python $draw_net $proto_file $image_file --rankdir=$rankdir