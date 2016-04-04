#!/usr/bin/env python 

acc_str = "1th: 0.966535, 2th: 0.973425, 3th: 0.96063, 4th: 0.950787, 5th: 0.818898, 6th: 0.801181, 7th: 0.992126, 8th: 0.837598, 9th: 0.793307, 10th: 0.992126, 11th: 0.969488, 12th: 0.844488, 13th: 0.988189, 14th: 0.998031"
# acc_str = "1th: 0.96752, 2th: 0.972441, 3th: 0.96063, 4th: 0.95374, 5th: 0.810039, 6th: 0.800197, 7th: 0.992126, 8th: 0.837598, 9th: 0.793307, 10th: 0.992126, 11th: 0.968504, 12th: 0.845472, 13th: 0.987205, 14th: 0.997047"

acc_str = "1th: 0.959646, 2th: 0.966535, 3th: 0.96063, 4th: 0.947835, 5th: 0.800197, 6th: 0.780512, 7th: 0.945866, 8th: 0.798228, 9th: 0.784449, 10th: 0.975394, 11th: 0.605315, 12th: 0.601378, 13th: 0.695866, 14th: 0.927165"

acc_str = "1th: 0.966535, 2th: 0.973425, 3th: 0.96063, 4th: 0.950787, 5th: 0.818898, 6th: 0.801181, 7th: 0.992126, 8th: 0.837598, 9th: 0.793307, 10th: 0.992126, 11th: 0.969488, 12th: 0.844488, 13th: 0.988189, 14th: 0.998031"

acc_str = "1th: 0.909449, 2th: 0.917323, 3th: 0.923228, 4th: 0.729331, 5th: 0.519685, 6th: 0.596457, 7th: 0.728346, 8th: 0.499016, 9th: 0.597441, 10th: 0.825787, 11th: 0.273622, 12th: 0.261811, 13th: 0.317913, 14th: 0.625"



acc_str = "1th: 0.956693, 2th: 0.962598, 3th: 0.96063, 4th: 0.948819, 5th: 0.797244, 6th: 0.773622, 7th: 0.942913, 8th: 0.78937, 9th: 0.778543, 10th: 0.974409, 11th: 0.610236, 12th: 0.603346, 13th: 0.687992, 14th: 0.925197"

acc_str = "1th: 0.937992, 2th: 0.958661, 3th: 0.95374, 4th: 0.952756, 5th: 0.802165, 6th: 0.766732, 7th: 0.933071, 8th: 0.780512, 9th: 0.771654, 10th: 0.974409, 11th: 0.605315, 12th: 0.599409, 13th: 0.685039, 14th: 0.924213"

acc_str = "1th: 0.948819, 2th: 0.958661, 3th: 0.955709, 4th: 0.945866, 5th: 0.784449, 6th: 0.749016, 7th: 0.936024, 8th: 0.783465, 9th: 0.773622, 10th: 0.974409, 11th: 0.605315, 12th: 0.57874, 13th: 0.684055, 14th: 0.91437"

# tmask 2000
acc_str = "1th: 0.951772, 2th: 0.959646, 3th: 0.96063, 4th: 0.950787, 5th: 0.771654, 6th: 0.746063, 7th: 0.918307, 8th: 0.745079, 9th: 0.772638, 10th: 0.974409, 11th: 0.576772, 12th: 0.580709, 13th: 0.670276, 14th: 0.912402"

# no torso mask 20000
acc_str = "1th: 0.94685, 2th: 0.942913, 3th: 0.950787, 4th: 0.937008, 5th: 0.773622, 6th: 0.730315, 7th: 0.938976, 8th: 0.768701, 9th: 0.708661, 10th: 0.981299, 11th: 0.600394, 12th: 0.605315, 13th: 0.67126, 14th: 0.889764"


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