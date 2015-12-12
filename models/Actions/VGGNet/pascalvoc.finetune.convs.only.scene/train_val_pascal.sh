# When use different environment, all you need to do is just to commend or umcomment 
# DDK
export LD_LIBRARY_PATH=/home/black/caffe/cuddn.env/v2/lib/:$LD_LIBRARY_PATH
# Zhu.Jin
# export LD_LIBRARY_PATH=/home/liangzhujin/env/common/cudnn/v2/lib/:$LD_LIBRARY_PATH


# ##############################################################################
# 
# Here use relative path
ProjectDir="../../../../"
DataDir="data/Actions/PascalVoc2012/"
CmdDir="tools"

# CaffeNet or VGGNet
SubProtoDir="VGGNet/"
ProtoDir="models/Actions/"
ProtoDir=$ProtoDir$SubProtoDir
# Get the *.prototxt files
SolverProtoName="pascalvoc.finetune.convs.only.scene/"
SolverProtoDir=$ProtoDir$SolverProtoName
solvername="pascal_action_solver.prototxt"
# e.g. "models/Actions/VGGNet/pascalvoc.finetune.convs.only.scene/pascal_action_solver.prototxt
# For training
solverpath=$SolverProtoDir$solvername
# For testing
defname="pascal_action_test.prototxt"
defpath=$SolverProtoDir$defname



# dataset to train on, the format of `imdb_name` is `rcnn_<datatype>_<split>, 
# where datatype is the name of action dataset, like:
# 	PascalVoc2012, Willowactions, Stanford, MPII
# where split indicates the train/val/trainval/test.
# 	rcnn_PascalVoc2012_train
train_imdbname="rcnn_PascalVoc2012_train"
test_imdbname="rcnn_PascalVoc2012_val"

# e.g. "VGGNet/pascalvoc.finetune.convs.only.scene/"
exper_name=$SubProtoDir$SolverProtoName

# 
echo
echo "######################################"
echo
echo "Usage: train_val_pascal.sh [re_iter]"
echo
echo "######################################"
echo
sleep_time=5
if [ ! -n "$1" ] ;then
	re_iter=0
	# Pretrained model directory
	# Be cafeful, the pretrained model directory may not different
	ModelDir="data/imagenet_models/"
	weightsname="VGG16.v2.caffemodel"
	# e.g. "data/imagenet_models/VGG16.v2.caffemodel"
	weightspath=$ModelDir$weightsname
	echo "weightspath: " $weightspath
	echo "re_iter: " $re_iter
	sleep $sleep_time
else
	re_iter=$1
	ModelDir="action_output/PascalVoc2012/"
	ModelDir=$ModelDir$train_imdbname"/"
	weightsname="vggnet_pascal_action_iter_"$re_iter".caffemodel"
	# e.g. "action_output/PascalVoc2012/rcnn_PascalVoc2012_train/VGGNet/pascalvoc.finetune.convs.only.scene"
	weightspath=$ModelDir$exper_name$weightsname
	echo "weightspath: " $weightspath
	echo "re_iter: " $re_iter
	sleep $sleep_time
fi


# ##############################################################################
# Used for the stored models directory

# The configuration file
cfg_filename="pascal_config.yml"
cfg_filepath=$SolverProtoDir$cfg_filename


# ##############################################################################
# Log

LogPath="output/action.logs/"
LogPath=$LogPath
cur_file_prefix="pascal_action_train_val_"
cur_log_filename=$(date -d "today" +"%Y.%m.%d.%H.%M.%S")
log_imdb_name="rcnn_PascalVoc2012_trainval"
log_filepath=$LogPath$log_imdb_name/$exper_name
log_filename=$cur_file_prefix$cur_log_filename.log
log_file=$log_filepath$log_filename


# ##############################################################################
# Run

cd $ProjectDir

mkdir -p $log_filepath

python $CmdDir/train_test_action_net.py \
		--gpu 0 \
		--re_iter $re_iter \
		--solver $solverpath \
		--weights $weightspath \
		--def $defpath \
		--exper_name $exper_name \
		--train_imdb $train_imdbname \
		--test_imdb $test_imdbname \
		--cfg $cfg_filepath \
		# 2>&1 | tee -a $log_file