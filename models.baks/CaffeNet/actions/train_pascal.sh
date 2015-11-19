TOOLSDIR="/home/black/caffe/fast-rcnn-action/"
PROTODIR="models/CaffeNet/actions/"
PROTODIR=$TOOLSDIR$PROTODIR
DATADIR="data/Actions/PascalVoc2012/"
MODELDIR="data/imagenet_models/"
CMDDIR="tools"
CMDDIR=$TOOLSDIR$CMDDIR

solvername="pascal_action_solver.prototxt"
solverpath=$PROTODIR$solvername

weightsname="CaffeNet.v2.caffemodel"
weightspath=$MODELDIR$weightsname

# dataset to train on, the format of `imdb_name` is `rcnn_<datatype>_<split>, 
# where datatype is the name of action dataset, like:
# PascalVoc2012, Willowactions, Stanford, MPII
# where split indicates the train/val/trainval/test.
# rcnn_PascalVoc2012_train
imdbname="rcnn_PascalVoc2012_train"

# prefix
# lof file
# log path
log_path="action_output/logs/"
log_path=$TOOLSDIR$log_path
mkdir -p $log_path
cur_file_prefix="pascal_action_"
cur_log_filename=$(date -d "today" +"%Y%m%d%H%M%S")
log_filename=$log_path/$cur_file_prefix$cur_log_filename.log


# execute
cd 
cd $TOOLSDIR
python $CMDDIR/train_action_net.py \
		--gpu 0 \
		--solver $solverpath \
		--weights $weightspath
		--imdb $imdbname \

# $1 2>&1 | tee -a $log_filename