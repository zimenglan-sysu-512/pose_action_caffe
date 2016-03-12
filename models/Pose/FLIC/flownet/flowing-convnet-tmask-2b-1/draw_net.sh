python_dir="../../../../../caffe/python/"
cd $python_dir

rankdir="TB"
draw_net="draw_net.py"
proto_file="../../models/Pose/FLIC/flownet/flowing-convnet-tmask-2b-1/train_val.prototxt"
image_file="../../models/Pose/FLIC/flownet/flowing-convnet-tmask-2b-1/tb_train_val.jpg"

python $draw_net $proto_file $image_file --rankdir=$rankdir