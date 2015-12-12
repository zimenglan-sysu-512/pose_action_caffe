TOOLSDIR="/home/black/caffe/fast-rcnn-action/"
PROTODIR="models/CaffeNet/actions/"
PROTODIR=$TOOLSDIR$PROTODIR
DATADIR="data/Actions/PascalVoc2012/"
MODELDIR="data/imagenet_models/"
CMDDIR="tools"
CMDDIR=$TOOLSDIR$CMDDIR

solvername="solver.prototxt"
solverpath=$PROTODIR$solvername

weightsname="CaffeNet.v2.caffemodel"
weightspath=$MODELDIR$weightsname

# dataset to train on, the format of `imdb_name` is `rcnn_<datatype>_<split>, 
# where datatype is the name of action dataset, like:
# PascalVoc2012, Willowactions, Stanford, MPII
# where split indicates the train/val/trainval/test.
# rcnn_PascalVoc2012_train
imdbname="rcnn_PascalVoc2012_train"


# execute
cd 
cd $TOOLSDIR
python $CMDDIR/train_action_net.py \
		--gpu 0 \
		--solver $solverpath \
		--weights $weightspath
		--imdb $imdbname