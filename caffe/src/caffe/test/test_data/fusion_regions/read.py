cls_sr_inds_file = "cls_sr_inds.log"
primary_regions_file = "primary_regions.log"
secondary_regions_file = "secondary_regions.log"

batch_size = 28
ims_per_batch = 2
n_classes = 11
n_regions = 5

fh = open(cls_sr_inds_file)
cls_sr_inds = fh.readlines()
fh.close()
cls_sr_inds = [csi.strip().split() for csi in cls_sr_inds]

fh = open(primary_regions_file)
primary_regions = fh.readlines()
fh.close()
primary_regions = [pr.strip() for pr in primary_regions]
primary_regions = [pr.split() for pr in primary_regions]


fh = open(secondary_regions_file)
secondary_regions = fh.readlines()
fh.close()
secondary_regions = [sr.strip() for sr in secondary_regions]


cls_regions = []
for csi in cls_sr_inds:
	print csi
	for c in csi:
		c = int(c)
		cls_regions.append(secondary_regions[c])

print 
print "****************************************"
print 
for cr in cls_regions:
	print cr.strip()
print len(cls_regions)

print 
print "****************************************"
print 

# secondary_regions = [sr.split() for sr in secondary_regions]
cls_regions = [cr.split() for cr in cls_regions]

fusion_regions = []
for cr in cls_regions:
	for pr in primary_regions:
		if cr[0] == pr[0]:
			str_info = cr[0]
			str_info = str_info + " " + str(min(int(cr[1]), int(pr[1])))
			str_info = str_info + " " + str(min(int(cr[2]), int(pr[2])))
			str_info = str_info + " " + str(max(int(cr[3]), int(pr[3])))
			str_info = str_info + " " + str(max(int(cr[4]), int(pr[4])))
			str_info = str_info + "\n"
			print str_info.strip()