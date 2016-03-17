python_dir="../../../../../caffe/python/"
cd $python_dir

rankdir="TB"
draw_net="draw_net.py"
proto_file="../../models/Pose2/flic/ddk/flo_net_no-tmask_1b/train.pt"
image_file="../../models/Pose2/flic/ddk/flo_net_no-tmask_1b/net.jpg"

python $draw_net $proto_file $image_file --rankdir=$rankdir