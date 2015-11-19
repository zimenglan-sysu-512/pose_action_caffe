python_dir="../../../../../caffe-fast-rcnn/python/"
cd $python_dir

rankdir="TB"
draw_net="draw_net.py"
proto_file="../../models/Pose/FLIC/baselines/lecun-8x-2b-addc4_2-sd5_5-marks/train_val.prototxt"
image_file="../../models/Pose/FLIC/baselines/lecun-8x-2b-addc4_2-sd5_5-marks/tb_train_val.jpg"

python $draw_net $proto_file $image_file --rankdir=$rankdir