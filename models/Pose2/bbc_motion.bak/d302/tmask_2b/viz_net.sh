python_dir="../../../../../caffe/python/"
cd $python_dir

rankdir="TB"
draw_net="draw_net.py"
proto_file="../../models/Pose2/bbc_motion/d302/tmask_2b/train.pt"
image_file="../../models/Pose2/bbc_motion/d302/tmask_2b/net.jpg"

python $draw_net $proto_file $image_file --rankdir=$rankdir