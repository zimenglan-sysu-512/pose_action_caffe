import os
import sys
import cv2

def images2vedio(in_dir, out_dir, out_name, fps=2):
  im_names = os.listdir(in_dir)
  im_names.sort()
  im_paths = [in_dir + im_name.strip() for im_name in im_names]

  out_path = out_dir + out_name
  # error: 'module' object has no attribute VideoWriter_fourcc
  # fourcc = cv2.VideoWriter_fourcc(*'avi')	# error
  fourcc = cv2.cv.CV_FOURCC(*'XVID')
  out = cv2.VideoWriter(out_path, fourcc, fps, (640,480))

  for im_path in im_paths:
  	im = cv2.imread(im_path)
  	out.write(im)
  	if (cv2.waitKey(1) & 0xFF) == ord('q'):
  		break
  out.release()

if __name__ == '__main__':

	in_dir = "/home/ddk/malong/dataset/person.torso/demo/vision/mude.images.pose1/"
	out_name = "demo.avi"
	out_dir = "/home/ddk/malong/dataset/person.torso/demo/vision/mude.images.pose1/"

	images2vedio(in_dir, out_dir, out_name)