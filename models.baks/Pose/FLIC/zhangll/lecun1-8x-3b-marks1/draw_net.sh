python_dir="../../../../../caffe/python/"
cd $python_dir

rankdir="TB"
draw_net="draw_net.py"
proto_file="../../models/Pose/FLIC/zhangll/lecun1-8x-3b-marks1/train_val.prototxt"
image_file="../../models/Pose/FLIC/zhangll/lecun1-8x-3b-marks1/tb_train_val.jpg"

python $draw_net $proto_file $image_file --rankdir=$rankdir