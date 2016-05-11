#!/usr/bin/env python 


# example
acc_str = "1th: 0.904528, 2th: 0.902559, 3th: 0.890748, 4th: 0.719488, 5th: 0.465551, 6th: 0.563976, 7th: 0.700787, 8th: 0.491142, 9th: 0.550197, 10th: 0.806102, 11th: 0.260827, 12th: 0.276575, 13th: 0.352362, 14th: 0.557087"


# #############################################################
acc = acc_str.strip().split(",")
acc = [float(s.strip().split(":")[1].strip()) for s in acc]

partname = [
	"reye", "leye", "nose",	# 0, 1, 2
	"rsho", "relb", "rwri",	# 3, 4, 5
	"lsho", "lelb", "lwri",	# 6, 7, 8 
	"neck", "rhip", "lhip", "btorso", "torso"	# 9, 10, 11, 12, 13
]

s = sum(acc)
l = len(acc)
ave = s / l
print
for idx in range(0, l):
	print "%s: %s" % (partname[idx], acc[idx])
print
print "**********************"
print "sum:", s
print "len:", l
print "ave:", ave
print

need_idxs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11]
acc1 = [acc[idx] for idx in need_idxs]
s = sum(acc1)
l = len(acc1)
ave = s / l
print "**********************"
print "sum:", s
print "len:", l
print "ave:", ave
print

need_idxs = [0, 1, 2, 3, 4, 5, 6, 7, 8]
acc1 = [acc[idx] for idx in need_idxs]
s = sum(acc1)
l = len(acc1)
ave = s / l
print "**********************"
print "sum:", s
print "len:", l
print "ave:", ave
print

need_idxs = [2, 3, 4, 5, 6, 7, 8,]
acc2 = [acc[idx] for idx in need_idxs]
s = sum(acc2)
l = len(acc2)
ave = s / l
print "**********************"
print "sum:", s
print "len:", l
print "ave:", ave
print


print "head:",  acc[2]
print "shoud:", (acc[3] + acc[6]) / 2.0
print "elbow:", (acc[4] + acc[7]) / 2.0
print "wrist:", (acc[5] + acc[8]) / 2.0



