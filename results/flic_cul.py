#!/usr/bin/env python 

acc_str = "1th: 0.947835, 2th: 0.956693, 3th: 0.958661, 4th: 0.948819, 5th: 0.790354, 6th: 0.760827, 7th: 0.937992, 8th: 0.775591, 9th: 0.772638, 10th: 0.974409, 11th: 0.605315, 12th: 0.580709, 13th: 0.684055, 14th: 0.917323"
acc_str = "1th: 0.949803, 2th: 0.956693, 3th: 0.945866, 4th: 0.948819, 5th: 0.790354, 6th: 0.741142, 7th: 0.941929, 8th: 0.792323, 9th: 0.761811, 10th: 0.973425, 11th: 0.607283, 12th: 0.609252, 13th: 0.67815, 14th: 0.917323"
acc_str = "1th: 0.950787, 2th: 0.96752, 3th: 0.955709, 4th: 0.940945, 5th: 0.794291, 6th: 0.754921, 7th: 0.937008, 8th: 0.798228, 9th: 0.765748, 10th: 0.971457, 11th: 0.600394, 12th: 0.586614, 13th: 0.686024, 14th: 0.915354"
acc_str = "1th: 0.959646, 2th: 0.966535, 3th: 0.96063, 4th: 0.947835, 5th: 0.800197, 6th: 0.780512, 7th: 0.945866, 8th: 0.798228, 9th: 0.784449, 10th: 0.975394, 11th: 0.605315, 12th: 0.601378, 13th: 0.695866, 14th: 0.927165"

acc = acc_str.strip().split(",")
acc = [float(s.strip().split(":")[1].strip()) for s in acc]
# print acc

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