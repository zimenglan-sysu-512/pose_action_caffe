#!/usr/bin/env python 

'''
0:  head
1:  neck
2:  top torso
3:  middle torso
4:  bottom torso
5:  right shoulder
6:  right elbow
7:  right wrist
8:  right hand
9:  left shoulder
10: left elbow
11: left wrist
12: left hand
13: right hip
14: right knee
15: right foot
16: left hip
17: left knee
18: left foot
'''


# example
acc_str = "1th: 0.976508, 2th: 0.983877, 3th: 0.967389, 4th: 0.939228, 5th: 0.810389, 6th: 0.948348, 7th: 0.710513, 8th: 0.662654, 9th: 0.764208, 10th: 0.959729, 11th: 0.707814, 12th: 0.573794, 13th: 0.668126, 14th: 0.843365, 15th: 0.904574, 16th: 0.945356, 17th: 0.789961, 18th: 0.923616, 19th: 0.954038"



# #############################################
acc = acc_str.strip().split(",")
acc = [float(s.strip().split(":")[1].strip()) for s in acc]


parts_names1 = [
	"head", "neck", "t_torso", "m_torso", "n_torso", # 0, 1, 2, 3, 4
	"r-sho", "r-elb", "r-wri", "r-hand",             # 5, 6, 7, 8
	"l-sho", "l-elb", "l-wri", "l-hand",             # 9, 10, 11, 12
	"r-hip", "r-knee", "r-foot",										 # 13, 14, 15
	"l-hip", "l-knee", "l-foot",										 # 16, 17, 18
]

s = sum(acc)
l = len(acc)
ave = s / l
print
for idx in range(0, l):
	print "%s: %s" % (parts_names1[idx], acc[idx])
print
print "**********************"
print "sum:", s
print "len:", l
print "ave:", ave
print
print "\n\n"


# ################################################################################


parts_names2 = [
	"head",     "neck",  "torso", # 0, 1, 2,
	"shoulder", "elbow", "hand",  # 3, 4, 5
	"hip",      "knee",  "foot"		# 7, 8, 9
]
nIdxs = [[0], [1], [2,3,4], [5,9], [6,10], [7,8,11,12], [13,16], [14,17], [15,18]]
acc1  = []
for j in nIdxs:
	s = 0
	for j2 in j:
		s2 = acc[j2]
		s += s2
	s = s / len(j)
	acc1.append(s)

s     = sum(acc1)
l     = len(acc1)
ave   = s / l

for idx in range(0, l):
	print "%s: %s" % (parts_names2[idx], acc1[idx])
print
print "**********************"
print "sum:", s
print "len:", l
print "ave:", ave
print
