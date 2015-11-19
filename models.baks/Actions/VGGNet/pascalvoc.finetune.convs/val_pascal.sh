export LD_LIBRARY_PATH=/home/black/caffe/cuddn.env/v2/lib/:$LD_LIBRARY_PATH

TOOLSDIR="/home/black/caffe/fast-rcnn-action/"
DATADIR="data/Actions/PascalVoc2012/"
CMDDIR="tools"
CMDDIR=$TOOLSDIR$CMDDIR

PROTODIR="models/Actions/VGGNet/"
# 
SUBPROTODIR="pascalvoc.finetune.convs/"
PROTODIR=$TOOLSDIR$PROTODIR$SUBPROTODIR
solvername="pascal_action_test.prototxt"
solverpath=$PROTODIR$solvername

EXPERNAME="vggnet."
EXPERNAME=$EXPERNAME$SUBPROTODIR
MODELDIR="action_output/PascalVoc2012/rcnn_PascalVoc2012_train/"
weightsname="vggnet_pascal_action_iter_28000.caffemodel"
weightspath=$MODELDIR$EXPERNAME$weightsname

# dataset to train on, the format of `imdb_name` is `rcnn_<datatype>_<split>, 
# where datatype is the name of action dataset, like:
# PascalVoc2012, Willowactions, Stanford, MPII
# where split indicates the train/val/trainval/test.
# rcnn_PascalVoc2012_train
imdbname="rcnn_PascalVoc2012_val"
exper_name=$EXPERNAME
has_show_secondary_region=0


log_path="output/action.logs"
log_path=$TOOLSDIR$log_path
mkdir -p $log_path
cur_file_prefix="pascal_action_val_"
cur_log_filename="accuracy"
log_filename=$log_path/$imdbname/$exper_name
mkdir -p $log_filename
log_filename=$log_filename$cur_file_prefix$cur_log_filename.log


# execute
cd 
cd $TOOLSDIR
python $CMDDIR/test_action_net.py \
		--gpu 0 \
		--def $solverpath \
		--net $weightspath \
		--imdb $imdbname \
		--has_show_secondary_region $has_show_secondary_region \
		2>&1 | tee -a $log_filename