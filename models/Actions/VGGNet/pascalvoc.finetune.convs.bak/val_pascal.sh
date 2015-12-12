TOOLSDIR="/home/black/caffe/fast-rcnn-action/"
PROTODIR="models/Actions/VGGNet/pascalvoc.finetune.convs/"
PROTODIR=$TOOLSDIR$PROTODIR
DATADIR="data/Actions/PascalVoc2012/"
MODELDIR="action_output/PascalVoc2012/rcnn_PascalVoc2012_train/vggnet.finetune.convs/"
MODELDIR=$TOOLSDIR$MODELDIR
CMDDIR="tools"
CMDDIR=$TOOLSDIR$CMDDIR

solvername="pascal_action_test.prototxt"
solverpath=$PROTODIR$solvername

weightsname="vggnet_pascal_action_iter_32000.caffemodel"
weightspath=$MODELDIR$weightsname

# dataset to train on, the format of `imdb_name` is `rcnn_<datatype>_<split>, 
# where datatype is the name of action dataset, like:
# PascalVoc2012, Willowactions, Stanford, MPII
# where split indicates the train/val/trainval/test.
# rcnn_PascalVoc2012_train
imdbname="rcnn_PascalVoc2012_val"
exper_name="vggnet.finetune.convs/"


# execute
cd 
cd $TOOLSDIR
python $CMDDIR/test_action_net.py \
		--gpu 0 \
		--def $solverpath \
		--net $weightspath \
		--imdb $imdbname \
		--exper_name $exper_name