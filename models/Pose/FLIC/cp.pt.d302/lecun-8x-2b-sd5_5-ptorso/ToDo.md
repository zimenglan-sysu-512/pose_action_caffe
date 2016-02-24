Note:
	1 hard negatives, use top k
	2 two banks, with conv4_2
	3 predicted torso bbox using full connection layers (regression)
	4 thres: 0.273
	5 prob_num: 11
	6 predicted torso bbox will be used for creating torso masks and then concated with conv4 for pose net

batch size increases, the iteration for devergence decreases, and accuracy 
(PDJ@0.1) increases

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