npython_dir="../../../../../caffe/python/"
cd $python_dir

rankdir="TB"
draw_net="draw_net.py"
proto_file="../../models/Pose/FLIC/liangzj/lecun1-8x-2b-addc4_2-marks/train_val.prototxt"
image_file="../../models/Pose/FLIC/liangzj/lecun1-8x-2b-addc4_2-marks/tb_train_val.jpg"

python $draw_net $proto_file $image_file --rankdir=$rankdir