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
# e.g. "models/Actions/CaffeNet/pascalvoc.finetune.convs.only.scene/pascal_action_solver.prototxt
solverpath=$SolverProtoDir$solvername

# Pretrained model directory
# Be cafeful, the pretrained model directory may not different
ModelDir="data/imagenet_models/"
weightsname="VGG16.v2.caffemodel"
# e.g. "data/imagenet_models/CaffeNet.v2.caffemodel"
weightspath=$ModelDir$weightsname


# ##############################################################################
# Used for the stored models directory

# dataset to train on, the format of `imdb_name` is `rcnn_<datatype>_<split>, 
# where datatype is the name of action dataset, like:
# 	PascalVoc2012, Willowactions, Stanford, MPII
# where split indicates the train/val/trainval/test.
# 	rcnn_PascalVoc2012_train
imdbname="rcnn_PascalVoc2012_train"

# The configuration file
cfg_filename="pascal_config.yml"
cfg_filepath=$SolverProtoDir$cfg_filename

# e.g. "CaffeNet/pascalvoc.finetune.convs.only.scene/"
exper_name=$SubProtoDir$SolverProtoName


# ##############################################################################
# Log

LogPath="output/action.logs/"
LogPath=$LogPath
cur_file_prefix="pascal_action_train_"
cur_log_filename=$(date -d "today" +"%Y.%m.%d.%H.%M.%S")
log_filepath=$LogPath$imdbname/$exper_name
mkdir -p $log_filepath
log_filename=$cur_file_prefix$cur_log_filename.log
log_file=$log_filepath$log_filename


# ##############################################################################
# Run

cd $ProjectDir
python $CmdDir/train_action_net.py \
		--gpu 0 \
		--solver $solverpath \
		--weights $weightspath \
		--imdb $imdbname \
		--exper_name $exper_name \
		--cfg $cfg_filepath \
		# 2>&1 | tee -a $log_file