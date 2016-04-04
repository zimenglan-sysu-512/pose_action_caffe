#!/usr/bin/env python 

acc_str = "1th: 0.958661, 2th: 0.956693, 3th: 0.956693, 4th: 0.939961, 5th: 0.779528, 6th: 0.775591, 7th: 0.941929, 8th: 0.795276, 9th: 0.750984, 10th: 0.980315, 11th: 0.602362, 12th: 0.604331, 13th: 0.683071, 14th: 0.900591"
acc_str = "1th: 0.961614, 2th: 0.964567, 3th: 0.952756, 4th: 0.937008, 5th: 0.799213, 6th: 0.761811, 7th: 0.989173, 8th: 0.830709, 9th: 0.733268, 10th: 0.992126, 11th: 0.963583, 12th: 0.823819, 13th: 0.98622, 14th: 0.998031"
# acc_str = "1th: 0.909449, 2th: 0.917323, 3th: 0.923228, 4th: 0.729331, 5th: 0.519685, 6th: 0.596457, 7th: 0.728346, 8th: 0.499016, 9th: 0.597441, 10th: 0.825787, 11th: 0.273622, 12th: 0.261811, 13th: 0.317913, 14th: 0.625"
acc_str = "1th: 0.909449, 2th: 0.917323, 3th: 0.923228, 4th: 0.729331, 5th: 0.519685, 6th: 0.596457, 7th: 0.728346, 8th: 0.499016, 9th: 0.597441, 10th: 0.825787, 11th: 0.273622, 12th: 0.261811, 13th: 0.317913, 14th: 0.625"
acc_str = "1th: 0.959646, 2th: 0.966535, 3th: 0.96063, 4th: 0.947835, 5th: 0.800197, 6th: 0.780512, 7th: 0.945866, 8th: 0.798228, 9th: 0.784449, 10th: 0.975394, 11th: 0.605315, 12th: 0.601378, 13th: 0.695866, 14th: 0.927165"
acc_str = "1th: 0.958661, 2th: 0.96752, 3th: 0.959646, 4th: 0.947835, 5th: 0.802165, 6th: 0.778543, 7th: 0.94685, 8th: 0.792323, 9th: 0.78937, 10th: 0.974409, 11th: 0.601378, 12th: 0.600394, 13th: 0.687992, 14th: 0.925197"
acc_str = "1th: 0.905512, 2th: 0.918307, 3th: 0.922244, 4th: 0.737205, 5th: 0.523622, 6th: 0.591535, 7th: 0.713583, 8th: 0.503937, 9th: 0.596457, 10th: 0.822835, 11th: 0.274606, 12th: 0.265748, 13th: 0.318898, 14th: 0.621063"
# acc_str = "1th: 0.906496, 2th: 0.915354, 3th: 0.922244, 4th: 0.739173, 5th: 0.517717, 6th: 0.595472, 7th: 0.709646, 8th: 0.500984, 9th: 0.599409, 10th: 0.818898, 11th: 0.276575, 12th: 0.264764, 13th: 0.329724, 14th: 0.625984"
acc_str = "1th: 0.909449, 2th: 0.916339, 3th: 0.922244, 4th: 0.729331, 5th: 0.518701, 6th: 0.596457, 7th: 0.728346, 8th: 0.501969, 9th: 0.598425, 10th: 0.824803, 11th: 0.274606, 12th: 0.261811, 13th: 0.317913, 14th: 0.624016"

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