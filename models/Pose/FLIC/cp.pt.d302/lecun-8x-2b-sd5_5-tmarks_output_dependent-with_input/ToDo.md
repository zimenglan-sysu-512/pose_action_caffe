Note:
	1 hard negatives, use top k
	2 two banks, with conv4_2 and spatial dropout
	3 predicted torso masks (2 categories -> bg v.s. torso, using softmax loss)
	4 thres: 0.273
	5 prob_num: 11
	6 use argmax layer to produce final torso mask (1 channel)
	7 final torso mask and input is concated with conv4 for pose net (channels: 512 + 3 + 1)

training method like faster-rcnn, torso mask net shares conv1~conv4 with heat map net, 
but torso mask is used for heat map net, meaning there are dependent. 
if torso mask net is trained, then the predicted torso mask will concat with some conv layer for pose net
	s1: train torso mask
	s2: train heat map
	s3: train torso mask net with fixing conv1~conv4
	s4: train heat  map  net with fixing conv1~conv4

batch size increases, the iteration for devergence decreases, and accuracy 
(PDJ@0.1) increases
12 -> 88.34
18 -> 88.92
24 -> ?

step1: 
 set large prob_num in heat map loss layer
 and lr_rate is 0.001
step2:
 when loss continues to decrease, but accuracy does not keep increasing,
 modify the prob_num to be smaller, e.g. 11 to 7
step3
 when loss continues to decrease, but accuracy does not keep increasing,
 modify the lr_rate, e.g. lr_rate /= 10

test:
	cd ../../../../../caffe/ && make -j8 && cd - && sh test.sh