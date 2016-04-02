python_dir="../../../../../caffe/python/"
cd $python_dir

rankdir="TB"
draw_net="draw_net.py"
proto_file="../../models/Pose2/bbc_motion/d302/fusnet_tmask_2b_ft_1/train.pt"
image_file="../../models/Pose2/bbc_motion/d302/fusnet_tmask_2b_ft_1/net.jpg"

python $draw_net $proto_file $image_file --rankdir=$rankdir