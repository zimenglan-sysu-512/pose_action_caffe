python_dir="../../../../../caffe/python/"
cd $python_dir

rankdir="TB"
draw_net="draw_net.py"
proto_file="../../models/Pose2/bbc_motion/ddk/fusnet_no-tmask_2b_1/train.pt"
image_file="../../models/Pose2/bbc_motion/ddk/fusnet_no-tmask_2b_1/net.jpg"

python $draw_net $proto_file $image_file --rankdir=$rankdir