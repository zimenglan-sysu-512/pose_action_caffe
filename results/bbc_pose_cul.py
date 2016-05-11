'''
head: 0
lsho: 1
lelb: 2
lwri: 3 
rsho: 4
relb: 5
rwri: 6
#####
lhip: 7
rhip: 8
'''


# example
acc_str = "1th: 0.968876, 2th: 1, 3th: 0.769076, 4th: 0.689759, 5th: 0.940763, 6th: 0.837349, 7th: 0.7249, 8th: 0.940763, 9th: 1"


# ##############################################################
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