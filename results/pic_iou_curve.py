#/usr/bin/env python

import os
import cv2
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score


disp_n    = 200
s_time    = 3
radius    = 3
thickness = 3
cls_color = (23, 119, 188)
colors    = [
		(0,     0, 255),
		(0,   255, 0),
		(255,   0, 0),
		(23,  119, 188), 
		(222,  12, 39), 
		(122, 212, 139), 
		(20,  198, 68), 
		(111,  12, 139), 
		(131, 112, 179), 
		(31,  211, 79), 
		(131, 121, 179), 
		(31,  121, 192), 
		(192,  21, 92), 
		(192,  21, 192), 
		(216, 121, 92), 
		(16,   11, 62), 
		(16,  111, 162), 
		(96,   46, 12), 
]
n_colors = len(colors)

def _mkdirs(path):
	if not os.path.isdir(path):
		os.makedirs(path)

# only one ground-truths for per image
def _read_gt(filepath):
	'''format: imgidx objidx bbox cls'''
	pd_dt = {}
	pd_c  = 0
	fh    = open(filepath)
	for line in fh.readlines():
		pd_c = pd_c + 1
		line = line.strip()
		info = line.split()
		assert len(info) >= 1
		imgidx, info = info[0], info[1:]
		assert len(info) == 6

		imgidx = imgidx.strip()
		objidx = info[0].strip()
		x1     = info[1].strip()
		y1     = info[2].strip()
		x2     = info[3].strip()
		y2     = info[4].strip()
		cls    = info[5].strip()

		objidx = int(objidx)
		assert objidx == 0

		x1 = int(x1)
		y1 = int(y1)
		x2 = int(x2)
		y2 = int(y2)

		if x1 > x2:
			x1, x2 = x2, x1
		if y1 > y2:
			y1, y2 = y2, y1

		pd_dt[imgidx] = [x1, y1, x2, y2]
	fh.close()
	assert pd_c == len(pd_dt.keys())

	return pd_dt

# multiple or one for prediction
def _read_pd(filepath, in_dire, is_in_dire=False):
	'''format: imgidx score bbox cls'''
	gt_dt = {}
	gt_c  = 0
	fh    = open(filepath)
	imgidxs = []
	for line in fh.readlines():
		gt_c = gt_c + 1
		line = line.strip()
		info = line.split()
		assert len(info) >= 1
		im_path, info = info[0], info[1:]
		assert len(info) == 6

		im_path = im_path.strip()
		score   = info[0].strip()
		x1      = info[1].strip()
		y1      = info[2].strip()
		x2      = info[3].strip()
		y2      = info[4].strip()
		cls     = info[5].strip()

		if is_in_dire:
			im_name = im_path[len(in_dire):]
		else:
			im_name = os.path.basename(im_path)
		imgidx  = im_name.strip().rsplit(".", 1)[0]
		imgidx = imgidx.strip()
		if imgidx in imgidxs:
			print imgidx, line
		imgidxs.append(imgidx)
		score = float(score)

		x1 = int(x1)
		y1 = int(y1)
		x2 = int(x2)
		y2 = int(y2)

		if x1 > x2:
			x1, x2 = x2, x1
		if y1 > y2:
			y1, y2 = y2, y1

		gt_dt[imgidx] = [x1, y1, x2, y2]
	fh.close()

	print len(imgidxs)
	print len(set(imgidxs))
	assert gt_c == len(gt_dt.keys()), "gt_c: %s, n_keys: %s" \
																	% (gt_c, len(gt_dt.keys()))
	return gt_dt

def _area(box):
	assert len(box) == 4

	w = box[2] - box[0] + 1
	h = box[3] - box[1] + 1
	a = w * h
	assert a >= 0
	return a

def _overlap(pd_box, gt_box):
	pa = _area(pd_box)
	ga = _area(gt_box)

	x1 = max(pd_box[0], gt_box[0])
	y1 = max(pd_box[1], gt_box[1])
	x2 = min(pd_box[2], gt_box[2])
	y2 = min(pd_box[3], gt_box[3])
	if x1 > x2 or y1 > y2:
		oa = 0
	else:
		oa = _area([x1, y1, x2, y2])

	return oa / (pa + ga - oa + 0.0)

def _iou(pd_file, gt_file, in_dire, is_in_dire=False):
	''''''
	pd_dt = _read_pd(pd_file, in_dire, is_in_dire=is_in_dire)
	gt_dt = _read_gt(gt_file)

	assert len(pd_dt.keys()) == len(gt_dt.keys())
	imgidxs = pd_dt.keys()
	imgidxs.sort()

	disp_c = 0
	ovs    = []
	for imgidx in imgidxs:
		disp_c += 1
		if disp_c % disp_n == 0:
			print "disp_c:", disp_c

		pd_box = pd_dt[imgidx]
		gt_box = gt_dt[imgidx]

		ov = _overlap(pd_box, gt_box)
		ovs.append(ov)

	if disp_c % disp_n != 0:
		print "disp_c:", disp_c
	print "\n\nDone.\n\n"

	return ovs

def _recall(ovs, thresolds):
	n_ovs = len(ovs) # n_examples
	n_thres = len(thresolds)

	precision = np.zeros(n_thres) # np.zeros((n_thres,), dtype=np.int)
	recall    = np.zeros(n_thres) # np.zeros((n_thres,), dtype=np.int)

	print recall.shape

	for j in xrange(n_thres):
		acc_c = 0
		thres = thresolds[j]
		for j2 in xrange(n_ovs):
			ov = ovs[j2]
			if ov > thres:
				acc_c += 1
		acc_c = acc_c / (n_ovs + 0.)
		precision[j] = acc_c
		recall[j]    = acc_c
	
	return recall
	
def _all_recall_pics(ovs_list, type_names, title, out_path=None, legend_loc="upper right"):
	'''Plot Precision-Recall curve'''
	plt.clf()
	plt.grid(True)
	plt.xlabel('IoU')
	plt.ylabel('Recall')
	# plt.ylim([0.0, 1.0])
	# plt.xlim([0.5, 1.0])
	
	n_dataset = len(ovs_list)
	assert n_dataset == len(type_names)

	thresolds = [j / 100.0 for j in xrange(50, 101, 1)]

	for j in xrange(n_dataset):
		ovs    = ovs_list[j]
		name   = type_names[j] 
		recall = _recall(ovs, thresolds)
		plt.plot(thresolds, recall, label=name)
		plt.xticks(np.arange(0.50, 1.01, 0.05))
		plt.yticks(np.arange(0.0, 1.01, 0.1))

	plt.title(title)
	plt.legend(loc=legend_loc)
	plt.savefig(out_path)
	if out_path is None:
		plt.show()
	else:
		plt.savefig(out_path)

def torso_run():
	''''''
	ovs_list   = []
	type_names = []
	out_path   = "/pathTo/../res.pics/torso.recall.png"

	## flic test
	pd_file    = "/pathTo/../dataset/FLIC/vision/flic_torso_test.txt"
	gt_file    = "/pathTo/../dataset/FLIC/labels/crop_test_torso_labels2.txt"
	in_dire    = "/pathTo/../dataset/FLIC/crop.images2/test/"
	is_in_dire = False
	type_names.append("FLIC Dataset")
	ovs        = _iou(pd_file, gt_file, in_dire, is_in_dire=is_in_dire)
	ovs_list.append(ovs)

	## bbc pose -> test & val
	pd_file    = "/pathTo/../dataset/bbc_pose/torso_masks/test_torso_results.txt"
	gt_file    = "/pathTo/../dataset/bbc_pose/labels/crop_test_torso.label"
	in_dire    = "/pathTo/../dataset/bbc_pose/crop.data/"
	is_in_dire = True
	type_names.append("BBC Pose Dataset")
	ovs        = _iou(pd_file, gt_file, in_dire, is_in_dire=is_in_dire)
	ovs_list.append(ovs)

	## kinect2
	pd_file    = "/pathTo/../dataset/Kinect2/torso_masks/test_torso_results.txt"
	gt_file    = "/pathTo/../dataset/Kinect2/labels/up.crop.color2_test_torso_l7.log"
	in_dire    = "/pathTo/../dataset/Kinect2/up.crop.color/"
	is_in_dire = False
	type_names.append("Kinect2 Dataset")
	ovs        = _iou(pd_file, gt_file, in_dire, is_in_dire=is_in_dire)
	ovs_list.append(ovs)
	
	# pic -> viz
	title = 'Recall for Torso Detection'
	_all_recall_pics(ovs_list, type_names, title, out_path=out_path)

def person_run():
	''''''
	ovs_list   = []
	type_names = []
	out_path   = "/pathTo/../res.pics/person.recall.png"

	## bbc pose -> test & val
	pd_file    = "/pathTo/../dataset/bbc_pose/test_person_results.txt"
	gt_file    = "/pathTo/../dataset/bbc_pose/labels/pbbox_test_cls.txt"
	in_dire    = "/pathTo/../dataset/bbc_pose/data/"
	is_in_dire = True
	type_names.append("BBC Pose Dataset")
	ovs        = _iou(pd_file, gt_file, in_dire, is_in_dire=is_in_dire)
	ovs_list.append(ovs)

	## kinect2
	pd_file    = "/pathTo/../dataset/Kinect2/test_person_results.txt"
	gt_file    = "/pathTo/../dataset/Kinect2/labels/up.color2.pbbox.test.log"
	in_dire    = "/pathTo/../dataset/Kinect2/up.color/"
	is_in_dire = False
	type_names.append("Kinect2 Dataset")
	ovs        = _iou(pd_file, gt_file, in_dire, is_in_dire=is_in_dire)
	ovs_list.append(ovs)
	
	# pic -> viz
	title = 'Recall for Person Detection'
	_all_recall_pics(ovs_list, type_names, title, out_path=out_path, legend_loc="lower left")



if __name__ == '__main__':
	''''''
	# torso_run()

	person_run()