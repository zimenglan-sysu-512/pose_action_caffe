acc_str = "1th: 0.934739, 2th: 1, 3th: 0.696787, 4th: 0.60241, 5th: 0.934739, 6th: 0.75, 7th: 0.763052, 8th: 0.932731, 9th: 1"
acc_str = "1th: 0.851406, 2th: 1, 3th: 0.47992, 4th: 0.433735, 5th: 0.851406, 6th: 0.545181, 7th: 0.545181, 8th: 0.841365, 9th: 1"

acc_str = "1th: 0.926707, 2th: 1, 3th: 0.599398, 4th: 0.512048, 5th: 0.930723, 6th: 0.696787, 7th: 0.611446, 8th: 0.926707, 9th: 1"
acc_str = "1th: 0.863454, 2th: 1, 3th: 0.531124, 4th: 0.444779, 5th: 0.837349, 6th: 0.569277, 7th: 0.491968, 8th: 0.850402, 9th: 1"
acc_str = "1th: 0.945783, 2th: 1, 3th: 0.742972, 4th: 0.614458, 5th: 0.933735, 6th: 0.777108, 7th: 0.702811, 8th: 0.934739, 9th: 1"
acc_str = "1th: 0.912651, 2th: 1, 3th: 0.643574, 4th: 0.522088, 5th: 0.89759, 6th: 0.665663, 7th: 0.601406, 8th: 0.894578, 9th: 1"

acc_str = "1th: 0.963855, 2th: 1, 3th: 0.728916, 4th: 0.697791, 5th: 0.950803, 6th: 0.697791, 7th: 0.726908, 8th: 0.937751, 9th: 1"
acc_str = "1th: 0.963855, 2th: 0.72992, 3th: 0.818273, 4th: 0.7249, 5th: 0.808233, 6th: 0.781124, 7th: 0.78012, 8th: 0.293173, 9th: 0.385542"
acc_str = "1th: 0.957831, 2th: 1, 3th: 0.754016, 4th: 0.695783, 5th: 0.939759, 6th: 0.782129, 7th: 0.762048, 8th: 0.935743, 9th: 1"

acc = acc_str.strip().split(",")
acc = [float(s.strip().split(":")[1].strip()) for s in acc]

partname = [
	"head",	# 0
	"lsho", # 1
	"lelb", # 2
	"lwri",	# 3 
	"rsho", # 4
	"relb", # 5
	"rwri",	# 6
	"lhip", # 7
	"rhip", # 8
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

need_idxs = [0, 1, 2, 3, 4, 5, 6]
acc1 = [acc[idx] for idx in need_idxs]
s = sum(acc1)
l = len(acc1)
ave = s / l
print "**********************"
print "sum:", s
print "len:", l
print "ave:", ave
print