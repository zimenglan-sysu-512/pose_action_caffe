Note:
	1 hard negatives, use top k
	2 two banks, use conv4_1 and conv4_2
	3 torso masks
	4 for kinect 2 dataset

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
	cd ../../../../../caffe/ && make -j8 && cd - && sh train.sh