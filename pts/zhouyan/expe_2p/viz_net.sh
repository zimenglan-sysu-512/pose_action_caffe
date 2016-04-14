python_dir="../../../caffe/python/"
cd $python_dir

rankdir="TB"
draw_net="draw_net.py"
proto_file="../../pts/zhouyan/expe_2p/train.pt"
image_file="../../pts/zhouyan/expe_2p/net.jpg"

python $draw_net $proto_file $image_file --rankdir=$rankdir