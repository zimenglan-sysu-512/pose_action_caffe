TOOLSDIR="/home/black/caffe/fast-rcnn-action/"
DATADIR="data/Actions/PascalVoc2012/"
CMDDIR="tools"
CMDDIR=$TOOLSDIR$CMDDIR

PROTODIR="models/Actions/CaffeNet/pascalvoc.freeze.convs/"
PROTODIR=$TOOLSDIR$PROTODIR
solvername="pascal_action_test.prototxt"
solverpath=$PROTODIR$solvername

MODELDIR="action_output/PascalVoc2012/rcnn_PascalVoc2012_train/caffenet.pascalvoc.freeze.convs/"
weightsname="caffenet_pascal_action_iter_100000.caffemodel"
weightspath=$MODELDIR$weightsname

# dataset to train on, the format of `imdb_name` is `rcnn_<datatype>_<split>, 
# where datatype is the name of action dataset, like:
# PascalVoc2012, Willowactions, Stanford, MPII
# where split indicates the train/val/trainval/test.
# rcnn_PascalVoc2012_train
imdbname="rcnn_PascalVoc2012_val"
exper_name="caffenet.pascalvoc.freeze.convs/"
has_show_secondary_region=0

# Log
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
		--exper_name $exper_name \
		--has_show_secondary_region $has_show_secondary_region \
		2>&1 | tee -a $log_filename
