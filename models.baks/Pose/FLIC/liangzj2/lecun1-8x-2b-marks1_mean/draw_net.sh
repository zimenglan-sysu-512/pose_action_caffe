python_dir="../../../../../caffe/python/"
cd $python_dir

rankdir="TB"
draw_net="draw_net.py"
proto_file="../../models/Pose/FLIC/liangzj2/lecun1-8x-2b-marks1_mean/train_val.prototxt"
image_file="../../models/Pose/FLIC/liangzj2/lecun1-8x-2b-marks1_mean/tb_train_val.jpg"

python $draw_net $proto_file $image_file --rankdir=$rankdir