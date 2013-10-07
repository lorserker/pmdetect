import cv2
import os
import os.path
import sys
import time

WIDTH = 640
HEIGHT = 480

def frame_generator(n, fps, dev=0):
	camera = cv2.VideoCapture(dev)
	for _ in range(n):
		retval, im = camera.read()
		yield im
		time.sleep(1.0 / fps)
	camera.release()

def record_webcam(outdir, n, fps, dev=0):
	for i, im in enumerate(frame_generator(n, fps, dev)):
		print '.',
		cv2.imwrite(os.path.join(outdir, 'webcam_%d.png' % i), im)

def frame_reader(folder):
	filenames = [fnm for fnm in os.listdir(folder) if fnm.endswith('.png')]
	ix_fnm = [(int(fnm[fnm.index('_') + 1 : fnm.index('.png')]), fnm) for fnm in filenames]
	ix_fnm.sort()
	for _, fnm in ix_fnm:
		yield cv2.imread(os.path.join(folder, fnm))


if __name__ == '__main__':
	record_webcam(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))