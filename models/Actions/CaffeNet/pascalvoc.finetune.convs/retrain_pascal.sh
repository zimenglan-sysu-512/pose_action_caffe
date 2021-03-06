TOOLSDIR="/home/black/caffe/fast-rcnn-action/"
DATADIR="data/Actions/PascalVoc2012/"
CMDDIR="tools"
CMDDIR=$TOOLSDIR$CMDDIR

PROTODIR="models/Actions/CaffeNet/pascalvoc.finetune.convs/"
PROTODIR=$TOOLSDIR$PROTODIR
solvername="pascal_action_solver.prototxt"
solverpath=$PROTODIR$solvername

MODELDIR="action_output/PascalVoc2012/rcnn_PascalVoc2012_train/caffenet.pascalvoc.finetune.convs/"
weightsname="caffenet_pascal_action_iter_28000.caffemodel"
weightspath=$MODELDIR$weightsname

# dataset to train on, the format of `imdb_name` is `rcnn_<datatype>_<split>, 
# where datatype is the name of action dataset, like:
# PascalVoc2012, Willowactions, Stanford, MPII
# where split indicates the train/val/trainval/test.
# rcnn_PascalVoc2012_train
imdbname="rcnn_PascalVoc2012_train"
exper_name="caffenet.pascalvoc.finetune.convs/"
re_iter="28000"

# Log
log_path="output/action.logs/"
log_path=$TOOLSDIR$log_path
mkdir -p $log_path
cur_file_prefix="pascal_action_retrain_"
cur_log_filename=$(date -d "today" +"%Y.%m.%d.%H.%M.%S")
log_filename=$log_path$imdbname/$exper_name
mkdir -p $log_filename
log_filename=$log_filename$cur_file_prefix$cur_log_filename.log

# execute
cd 
cd $TOOLSDIR
python $CMDDIR/train_action_net.py \
		--gpu 0 \
		--solver $solverpath \
		--weights $weightspath \
		--imdb $imdbname \
		--exper_name $exper_name \
		--re_iter $re_iter \
		2>&1 | tee -a $log_filename