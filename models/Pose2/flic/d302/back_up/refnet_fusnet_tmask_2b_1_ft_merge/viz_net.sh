python_dir="../../../../../caffe/python/"
cd $python_dir

rankdir="TB"
draw_net="draw_net.py"

proto_file="../../models/Pose2/flic/d302/refnet_fusnet_tmask_2b_1_ft_merge/train.pt"
image_file="../../models/Pose2/flic/d302/refnet_fusnet_tmask_2b_1_ft_merge/net.jpg"
python $draw_net $proto_file $image_file --rankdir=$rankdir

