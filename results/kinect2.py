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



# #############################################
# # motion - PCK@0.1
# # ref + fus + spa -> 7000 iteration
# # acc_str = "1th: 0.993069, 2th: 0.984096, 3th: 0.979791, 4th: 0.970088, 5th: 0.895747, 6th: 0.975633, 7th: 0.902532, 8th: 0.833662, 9th: 0.889181, 10th: 0.977019, 11th: 0.888378, 12th: 0.853578, 13th: 0.903042, 14th: 0.904064, 15th: 0.95754, 16th: 0.960458, 17th: 0.914934, 18th: 0.949296, 19th: 0.971474"

# # fus + spa -> 21000 iteration
# acc_str = "1th: 0.991975, 2th: 0.985847, 3th: 0.982783, 4th: 0.968337, 5th: 0.895601, 6th: 0.975852, 7th: 0.830598, 8th: 0.844678, 9th: 0.899905, 10th: 0.972496, 11th: 0.860582, 12th: 0.849274, 13th: 0.908003, 14th: 0.899467, 15th: 0.947326, 16th: 0.926169, 17th: 0.900489, 18th: 0.918655, 19th: 0.961407"

# # spa -> 25000 iteration
# acc_str = "1th: 0.993288, 2th: 0.986357, 3th: 0.979208, 4th: 0.961698, 5th: 0.906106, 6th: 0.96841, 7th: 0.79215, 8th: 0.830014, 9th: 0.867294, 10th: 0.964981, 11th: 0.814401, 12th: 0.850295, 13th: 0.896476, 14th: 0.906179, 15th: 0.92989, 16th: 0.902386, 17th: 0.910411, 18th: 0.914131, 19th: 0.945356"



# #############################################
# # no motion PCK@0.1
# # ref + fus + spa -> 0 iteration (actually use fus+spa 24000 iteration)
# acc_str = "1th: 0.995039, 2th: 0.984388, 3th: 0.982418, 4th: 0.966586, 5th: 0.873714, 6th: 0.954841, 7th: 0.888013, 8th: 0.855256, 9th: 0.884949, 10th: 0.924199, 11th: 0.912891, 12th: 0.83855, 13th: 0.892245, 14th: 0.872985, 15th: 0.94594, 16th: 0.963668, 17th: 0.891588, 18th: 0.934048, 19th: 0.97432"

# # fus + spa -> 0 iteration (actually use spa 5000 iteration)
# acc_str = "1th: 0.993872, 2th: 0.983731, 3th: 0.975487, 4th: 0.958634, 5th: 0.917123, 6th: 0.95433, 7th: 0.879478, 8th: 0.836069, 9th: 0.866346, 10th: 0.911213, 11th: 0.85015, 12th: 0.812213, 13th: 0.882688, 14th: 0.909025, 15th: 0.920624, 16th: 0.916101, 17th: 0.92668, 18th: 0.920624, 19th: 0.956081"

# # spa -> 3000 iteration
# acc_str = "1th: 0.993799, 2th: 0.982199, 3th: 0.973371, 4th: 0.949077, 5th: 0.903042, 6th: 0.953309, 7th: 0.816079, 8th: 0.807471, 9th: 0.852849, 10th: 0.920843, 11th: 0.841687, 12th: 0.751952, 13th: 0.837601, 14th: 0.894652, 15th: 0.919676, 16th: 0.927263, 17th: 0.913986, 18th: 0.917633, 19th: 0.957759"

# acc = acc_str.strip().split(",")
# acc = [float(s.strip().split(":")[1].strip()) for s in acc]



# #############################################
# # motion PCK@0.05
# # ref + fus + spa -> 7000 iteration
# acc_str = "1th: 0.963814, 2th: 0.865251, 3th: 0.873933, 4th: 0.84176, 5th: 0.611075, 6th: 0.794776, 7th: 0.677829, 8th: 0.524622, 9th: 0.6628, 10th: 0.812359, 11th: 0.697527, 12th: 0.534107, 13th: 0.68527, 14th: 0.611658, 15th: 0.756913, 16th: 0.715109, 17th: 0.597724, 18th: 0.660465, 19th: 0.75538"

# # fus + spa -> 13000 iteration
# acc_str = "1th: 0.963376, 2th: 0.866565, 3th: 0.861749, 4th: 0.829722, 5th: 0.622164, 6th: 0.817028, 7th: 0.614139, 8th: 0.533961, 9th: 0.66995, 10th: 0.810827, 11th: 0.659371, 12th: 0.532137, 13th: 0.68155, 14th: 0.610345, 15th: 0.696141, 16th: 0.695922, 17th: 0.577807, 18th: 0.615306, 19th: 0.731232"

# # spa -> 3000 iteration
# acc_str = "1th: 0.955059, 2th: 0.8662, 3th: 0.863136, 4th: 0.740716, 5th: 0.62873, 6th: 0.780988, 7th: 0.622456, 8th: 0.555337, 9th: 0.643102, 10th: 0.634931, 11th: 0.634566, 12th: 0.495294, 13th: 0.647625, 14th: 0.600715, 15th: 0.643905, 16th: 0.645364, 17th: 0.620778, 18th: 0.614868, 19th: 0.693733"

# acc = acc_str.strip().split(",")
# acc = [float(s.strip().split(":")[1].strip()) for s in acc]



# #############################################
# no motion PCK@0.05
# ref + fus + spa -> 7000 iteration
acc_str = "1th: 0.959947, 2th: 0.881302, 3th: 0.895674, 4th: 0.831546, 5th: 0.640695, 6th: 0.75093, 7th: 0.620121, 8th: 0.5843, 9th: 0.693368, 10th: 0.657474, 11th: 0.714598, 12th: 0.541475, 13th: 0.67834, 14th: 0.61589, 15th: 0.715547, 16th: 0.670241, 17th: 0.628949, 18th: 0.663748, 19th: 0.71044"

# fus + spa -> 3000 iteration
acc_str = "1th: 0.945283, 2th: 0.843292, 3th: 0.835267, 4th: 0.796381, 5th: 0.622675, 6th: 0.764719, 7th: 0.680382, 8th: 0.555993, 9th: 0.639673, 10th: 0.709856, 11th: 0.656307, 12th: 0.493179, 13th: 0.651711, 14th: 0.601153, 15th: 0.693879, 16th: 0.681841, 17th: 0.615671, 18th: 0.664697, 19th: 0.750565"

# spa -> 7000 iteration
acc_str = "1th: 0.95309, 2th: 0.840592, 3th: 0.846283, 4th: 0.729773, 5th: 0.635952, 6th: 0.772963, 7th: 0.601955, 8th: 0.5379, 9th: 0.662727, 10th: 0.64157, 11th: 0.576421, 12th: 0.487196, 13th: 0.655213, 14th: 0.618881, 15th: 0.616838, 16th: 0.617495, 17th: 0.60998, 18th: 0.612388, 19th: 0.665791"

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
