python_dir="../../../../../caffe/python/"
cd $python_dir

rankdir="TB"
draw_net="draw_net.py"
proto_file="../../models/Pose/FLIC/cp.pt.d302/lecun-8x-2b-sd5_5-tmarks_output_dependent/train_val.prototxt"
image_file="../../models/Pose/FLIC/cp.pt.d302/lecun-8x-2b-sd5_5-tmarks_output_dependent/tb_train_val.jpg"

python $draw_net $proto_file $image_file --rankdir=$rankdir