#/usr/bin/env python

import os
import cv2
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

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
		(222,  112, 89), 
		(122, 212, 139), 
		(20,  198, 168), 
		(111,  12, 139), 
		(131, 112, 179), 
		(31,  61, 79), 
		(131, 161, 179), 
		(31,  221, 192), 
		(192,  21, 92), 
		(192,  21, 192), 
		(216, 121, 92), 
		(16,   11, 62), 
		(16,  111, 162), 
		(96,   46, 12), 
]
colors2 = []
for color in colors:
	c2 = [c / 255.0 for c in color]
	colors2.append(tuple(c2))
colors = colors2
# print colors
n_colors = len(colors)

def mkdirs(path):
	if not os.path.isdir(path):
		os.makedirs(path)

def _parse_acc_str(res_str_file, per_lines=3):
	fh = open(res_str_file)
	res_strs = fh.readlines()
	fh.close()

	n_res_strs = len(res_strs)
	assert n_res_strs >= per_lines, res_str_file
	assert n_res_strs % per_lines == 0, res_str_file

	n_factors  = n_res_strs / per_lines
	factors    = np.zeros(n_factors)
	t_accuracy = np.zeros(n_factors)
	
	parts_acc_str  = res_strs[2]
	parts_acc      = parts_acc_str.strip().split(",")
	parts_acc      = [float(s.strip().split(":")[1].strip()) for s in parts_acc]
	n_parts        = len(parts_acc)
	p_accuracy     = np.zeros((n_parts, n_factors))

	for j in xrange(n_factors):
		j2    = j * per_lines
		fac   = res_strs[j2 + 0].strip()
		acc   = res_strs[j2 + 1].strip()
		p_acc = res_strs[j2 + 2].strip()

		fac   = fac.split(":")[1].strip()
		fac   = float(fac)

		acc   = acc.split(":")[1].strip()
		acc   = float(acc)

		p_acc = p_acc.split(",")
		p_acc = [float(s.strip().split(":")[1].strip()) for s in p_acc]
		assert len(p_acc) == n_parts

		factors[j]     = fac
		t_accuracy [j] = acc
		for j3 in xrange(n_parts):
			p_accuracy[j3][j] = p_acc[j3]
	
	t_accuracy2 = np.zeros(n_factors)
	for j in xrange(n_factors):
		s = sum(p_accuracy[:, j])
		s = s / n_parts
		t_accuracy2[j] = s

	return factors, t_accuracy2, p_accuracy

def _ave_sym_parts(acc, idxs):
	shape  						 = acc.shape
	n_parts, n_factors = shape
	
	n_parts2 = len(idxs)
	acc2     = np.zeros((n_parts2, n_factors))

	for j in xrange(n_factors):
		for j2 in xrange(n_parts2):
			s     = 0
			idxs2 = idxs[j2]
			for j3 in idxs2:
				s += acc[j3, j]
			s = s / len(idxs2)
			acc2[j2, j] = s

	return acc2


def pic_pck_curve(res_str_file, out_path, parts_names1, parts_names2, \
									xticks, yticks, x_title, y_tile, idxs=None, title=""):
	factors, t_accuracy, p_accuracy = _parse_acc_str(res_str_file)
	
	flag        = True
	p_accuracy2 = p_accuracy
	if idxs is not None:
		p_accuracy2 = _ave_sym_parts(p_accuracy, idxs)
		flag        = False
	parts_names = parts_names1
	if not flag:
		parts_names = parts_names2

	n_parts = p_accuracy2.shape[0]
	assert len(parts_names) == n_parts, "n_parts: %s" % (n_parts,)

	acc = [100 * ta for ta in t_accuracy]
	
	# init plt
	plt.clf()
	plt.grid(True)

	# plot t_accuracy
	plt.plot(factors, acc, label='average', color=colors[0], linewidth=1.5)
	plt.xlabel(x_title)
	plt.ylabel(y_title)

	if xticks is None:
		xticks = np.arange(0, factors[-1] + 0.01, 0.02)
	if yticks is None:
		yticks = np.arange(0, 101, 10)
	print xticks
	print yticks
	plt.xticks(xticks)
	plt.yticks(yticks)

	# plot parts
	for j in xrange(n_parts):
		acc = p_accuracy2[j, :]
		acc = [100 * c for c in acc]
		plt.plot(factors, acc, label=parts_names[j], \
						 color=colors[(j + 1) % n_colors], linewidth=1.5)
		plt.xlabel(x_title)
		plt.ylabel(y_title)
		plt.xticks(xticks)
		plt.yticks(yticks)

	plt.title(title)
	plt.legend(loc="lower right")
	# plt.show()
	plt.savefig(out_path)

if __name__ == '__main__':
	''''''
	im_ext = ".png"

	# # ##########################################################
	# # flic
	# res_str_file = "res_strs/flic_final_cascaded_8000_iter.txt"
	# parts_names1 = [
	# 	"r-eye",      "l-eye",   "nose",	  # 0, 1, 2
	# 	"r-shoulder", "r-elbow", "r-wrist",	# 3, 4, 5
	# 	"l-shoulder", "l-elbow", "l-wrist",	# 6, 7, 8 
	# 	# "neck", "rhip", "lhip", "btorso", "torso"	# 9, 10, 11, 12, 13
	# ]
	# # parts_names2 = [
	# # 	"head", "shoulder", "elbow", "hand",  # 0, 1, 2, 3
	# # ]
	# # idxs     = [[0,1,2], [3,6], [4,7], [5,8]]
	# parts_names2 = [
	# 	"r-eye",       "l-eye",  "nose",	  # 0, 1, 2
	# 	"r-shoulder", "r-elbow", "r-wrist",	# 3, 4, 5
	# 	"l-shoulder", "l-elbow", "l-wrist",	# 6, 7, 8 
	# ]
	# idxs     = [[0], [1], [2], [3], [4], [5], [6], [7], [8]]
	# out_dire = "pic.plot/flic/"
	# mkdirs(out_dire)
	# title    = "FLIC Dataset"
	# out_path = out_dire + title + im_ext
	# xticks = None # np.arange(0, factors[-1] + 0.01, 0.02)
	# yticks = np.arange(0, 101, 10)
	# x_title  = "Normalized distance"
	# y_title  = "Detection rate, %"
	# pic_pck_curve(res_str_file, out_path, parts_names1, parts_names2, \
	# 							xticks, yticks, x_title, y_title, idxs=idxs, title=title)

	# # ##########################################################
	# # kinect2 motion
	# res_str_file = "res_strs/kinect2_motion_final_cascaded_7000_iter.txt"
	# parts_names1 = [
	# 	"head", "neck", "t_torso", "m_torso", "n_torso", # 0, 1, 2, 3, 4
	# 	"r-sho", "r-elb", "r-wri", "r-hand",             # 5, 6, 7, 8
	# 	"l-sho", "l-elb", "l-wri", "l-hand",             # 9, 10, 11, 12
	# 	"r-hip", "r-knee", "r-foot",										 # 13, 14, 15
	# 	"l-hip", "l-knee", "l-foot",										 # 16, 17, 18
	# ]
	# parts_names2 = [
	# 	"head",     "neck",  "torso", # 0, 1, 2,
	# 	"shoulder", "elbow", "hand",  # 3, 4, 5
	# 	"hip",      "knee",  "foot"		# 7, 8, 9
	# ]
	# idxs     = [[0], [1], [2,3,4], [5,9], [6,10], [7,8,11,12], [13,16], [14,17], [15,18]]
	# out_dire = "pic.plot/kinect2_motion/"
	# mkdirs(out_dire)
	# title    = "Kinect2 Motion Dataset"
	# out_path = out_dire + title + im_ext
	# xticks = None # np.arange(0, factors[-1] + 0.01, 0.02)
	# yticks = np.arange(0, 101, 10)
	# x_title  = "Normalized distance"
	# y_title  = "Detection rate, %"
	# pic_pck_curve(res_str_file, out_path, parts_names1, parts_names2, \
	# 							xticks, yticks, x_title, y_title, idxs=idxs, title=title)

	# ##########################################################
	# flic
	res_str_file = "res_strs/bbc_pose_final_spa_fus_15000_iter.txt"
	parts_names1 = [
		"head",															# 0
		"l-shoulder", "l-elbow", "l-wrist", # 1, 2, 3
		"r-shoulder", "r-elbow", "r-wrist"  # 4, 5, 6
	]
	parts_names2 = [
		"head",															# 0
		"l-shoulder", "l-elbow", "l-wrist", # 1, 2, 3
		"r-shoulder", "r-elbow", "r-wrist"  # 4, 5, 6
	]
	idxs     = [[0], [1], [2], [3], [4], [5], [6]]
	out_dire = "pic.plot/bbc_pose/"
	mkdirs(out_dire)
	title    = "BBC Pose Dataset"
	out_path = out_dire + title + im_ext
	xticks = np.arange(0, 21, 2)
	yticks = None
	x_title  = "Distance from GT [px]"
	y_title  = "Accuracy [%]"
	pic_pck_curve(res_str_file, out_path, parts_names1, parts_names2, \
								xticks, yticks, x_title, y_title, idxs=idxs, title=title)