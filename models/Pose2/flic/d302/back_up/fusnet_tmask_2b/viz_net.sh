python_dir="../../../../../caffe/python/"
cd $python_dir

rankdir="TB"
draw_net="draw_net.py"

proto_file="../../models/Pose2/flic/d302/fusnet_tmask_2b/train_ft.pt"
image_file="../../models/Pose2/flic/d302/fusnet_tmask_2b/net_ft.jpg"
python $draw_net $proto_file $image_file --rankdir=$rankdir

proto_file="../../models/Pose2/flic/d302/fusnet_tmask_2b/train_fus.pt"
image_file="../../models/Pose2/flic/d302/fusnet_tmask_2b/net_fus.jpg"

python $draw_net $proto_file $image_file --rankdir=$rankdir