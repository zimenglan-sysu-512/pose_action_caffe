TOOLSDIR="/home/black/caffe/fast-rcnn-action/"
PROTODIR="models/CaffeNet/actions/"
PROTODIR=$TOOLSDIR$PROTODIR
DATADIR="data/Actions/PascalVoc2012/"
MODELDIR="action_output/default/rcnn_PascalVoc2012_train/"
CMDDIR="tools"
CMDDIR=$TOOLSDIR$CMDDIR

solvername="pascal_action_test.prototxt"
solverpath=$PROTODIR$solvername

weightsname="pascal_action_iter_40000.caffemodel"
weightspath=$MODELDIR$weightsname

# dataset to train on, the format of `imdb_name` is `rcnn_<datatype>_<split>, 
# where datatype is the name of action dataset, like:
# PascalVoc2012, Willowactions, Stanford, MPII
# where split indicates the train/val/trainval/test.
# rcnn_PascalVoc2012_train
imdbname="rcnn_PascalVoc2012_val"


# execute
cd 
cd $TOOLSDIR
python $CMDDIR/test_action_net.py \
		--gpu 0 \
		--def $solverpath \
		--net $weightspath \
		--imdb $imdbname