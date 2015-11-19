export LD_LIBRARY_PATH=/home/black/caffe/cuddn.env/v2/lib/:$LD_LIBRARY_PATH

# /home/black/caffe/fast-rcnn-action/models/Actions/VGGNet/pascalvoc.finetune.convs.only.scene/
TOOLSDIR="/home/black/caffe/fast-rcnn-action/"
DATADIR="data/Actions/PascalVoc2012/"
CMDDIR="tools"
CMDDIR=$TOOLSDIR$CMDDIR


PROTODIR="models/Actions/VGGNet/"
# 
SUBPROTODIR="pascalvoc.finetune.convs/"
PROTODIR=$TOOLSDIR$PROTODIR$SUBPROTODIR
solvername="pascal_action_solver.prototxt"
solverpath=$PROTODIR$solvername

MODELDIR="data/imagenet_models/"
weightsname="VGG16.v2.caffemodel"
weightspath=$MODELDIR$weightsname

# dataset to train on, the format of `imdb_name` is `rcnn_<datatype>_<split>, 
# where datatype is the name of action dataset, like:
# 	PascalVoc2012, Willowactions, Stanford, MPII
# where split indicates the train/val/trainval/test.
# 	rcnn_PascalVoc2012_train
imdbname="rcnn_PascalVoc2012_train"
cfg_filename="pascal_config.yml"
cfg_filepath=$PROTODIR$cfg_filename
exper_name="vggnet."
exper_name=$exper_name$SUBPROTODIR

# Log
log_path="output/action.logs/"
log_path=$TOOLSDIR$log_path
mkdir -p $log_path
cur_file_prefix="pascal_action_"
cur_log_filename=$(date -d "today" +"%Y.%m.%d.%H.%M.%S")
log_filename=$log_path$imdbname/$exper_name
mkdir -p $log_filename
log_filename=$log_filename$cur_file_prefix$cur_log_filename.log


cd 
cd $TOOLSDIR
python $CMDDIR/train_action_net.py \
		--gpu 0 \
		--solver $solverpath \
		--weights $weightspath \
		--imdb $imdbname \
		--exper_name $exper_name \
		--cfg $cfg_filepath \
		# 2>&1 | tee -a $log_filename
