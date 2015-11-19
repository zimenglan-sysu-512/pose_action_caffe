# When use different environment, all you need to do is just to commend or umcomment 
# DDK
export LD_LIBRARY_PATH=/home/black/caffe/cuddn.env/v2/lib/:$LD_LIBRARY_PATH
# Zhu.Jin
# export LD_LIBRARY_PATH=/home/liangzhujin/env/common/cudnn/v2/lib/:$LD_LIBRARY_PATH


# ##############################################################################
# 
# ProjectDir="/home/black/caffe/fast-rcnn-action/"
# ProjectDir="/home/liangzhujin/caffe/fast-rcnn-action/"
# Here use relative path
ProjectDir="../../../../"
DataDir="data/Actions/PascalVoc2012/"
CmdDir="tools"

# CaffeNet or VGGNet
SubProtoDir="CaffeNet/"
ProtoDir="models/Actions/"
ProtoDir=$ProtoDir$SubProtoDir
# Get the *.prototxt files
SolverProtoName="pascalvoc.finetune.convs.only.scene/"
SolverProtoDir=$ProtoDir$SolverProtoName
solvername="pascal_action_test.prototxt"
# e.g. "models/Actions/CaffeNet/pascalvoc.finetune.convs.only.scene/pascal_action_solver.prototxt
solverpath=$SolverProtoDir$solvername

# dataset to train on, the format of `imdb_name` is `rcnn_<datatype>_<split>, 
# where datatype is the name of action dataset, like:
# 	PascalVoc2012, Willowactions, Stanford, MPII
# where split indicates the train/val/trainval/test.
# 	rcnn_PascalVoc2012_train
train_imdbname="rcnn_PascalVoc2012_train"
re_iter="80"

# Pretrained model directory
# Be cafeful, the pretrained model directory may not different
ExperName=$SubProtoDir$SolverProtoName
ModelDir="action_output/PascalVoc2012/"
ModelDir=$ModelDir$train_imdbname"/"
weightsname="caffenet_pascal_action_iter_"$re_iter".caffemodel"
# e.g. "action_output/PascalVoc2012/rcnn_PascalVoc2012_train/CaffeNet/pascalvoc.finetune.convs.only.scene"
weightspath=$ModelDir$ExperName$weightsname
echo $weightspath


# ##############################################################################

imdbname="rcnn_PascalVoc2012_val"
# The configuration file
cfg_filename="pascal_config.yml"
cfg_filepath=$SolverProtoDir$cfg_filename

# e.g. "CaffeNet/pascalvoc.finetune.convs.only.scene/"
exper_name=$ExperName


# ##############################################################################
# Log

LogPath="output/action.logs/"
LogPath=$LogPath
cur_file_prefix="pascal_action_val_"
cur_log_filename=$(date -d "today" +"%Y.%m.%d.%H.%M.%S")
log_filepath=$LogPath$imdbname/$exper_name
log_filename=$cur_file_prefix$cur_log_filename.log
log_file=$log_filepath$log_filename


# ##############################################################################
# Run

cd $ProjectDir

mkdir -p $log_filepath

python $CmdDir/test_action_net.py \
		--gpu 0 \
		--def $solverpath \
		--net $weightspath \
		--imdb $imdbname \
		--exper_name $exper_name \
		--cfg $cfg_filepath \
		# 2>&1 | tee -a $log_filename