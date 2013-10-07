import sys
import cv2
import os
import os.path
import numpy as np
import scipy.ndimage
import capture

from scipy.stats import norm

WINDOW_SIZE = 5

def to_greyscale(im):
	im_grey = np.zeros(im.shape[:2])
	im_grey = im[:, :, 0] * 0.3 + im[:, :, 1] * 0.5 + im[:, :, 2] * 0.2
	return im_grey

def blur(im, sigma):
	return scipy.ndimage.gaussian_filter(im, sigma, mode='wrap')


class PixModel(object):

	def __init__(self, shape, window):
		self.N = window
		self.mu = 128.0 * np.ones(shape)
		self.sigma = 10.0 * np.ones(shape)
		self.alpha_mu = 2.0 / (self.N + 1)
		self.alpha_sigma = 2.0 / (10 * self.N + 1)

	def update(self, img):
		new_mu = np.zeros(self.mu.shape)
		new_sigma = np.zeros(self.sigma.shape)
		new_mu = self.alpha_mu * img + (1 - self.alpha_mu) * self.mu
		new_sigma = np.sqrt(self.alpha_sigma * (img - new_mu)**2 
			+ (1 - self.alpha_sigma) * self.sigma**2)
		self.mu = new_mu
		self.sigma = new_sigma

	def get_p_change(self, img):
		d = self.mu - np.fabs(img - self.mu)
		p_change = 1 - 2 * norm.cdf(d, loc=self.mu, scale=self.sigma)
		return p_change

def motion_detect_demo(in_folder, out_frame, out_current, out_pixmodel, out_pchange, out_changemask):
	i = 0
	pixmodel = PixModel((capture.HEIGHT, capture.WIDTH), WINDOW_SIZE)
	for frame in capture.frame_reader(in_folder):		
		cv2.imwrite(os.path.join(out_frame, '%d.png' % i), frame)
		im = blur(to_greyscale(frame), 10)
		if i == 0:
			pixmodel.mu = im
		cv2.imwrite(os.path.join(out_current, '%d.png' % i), im)
		cv2.imwrite(os.path.join(out_pixmodel, '%d.png' % i), pixmodel.mu)
		p_chg = pixmodel.get_p_change(im)
		cv2.imwrite(os.path.join(out_pchange, '%d.png' % i), 255*p_chg)
		p_chg_98 = p_chg.copy()
		p_chg_98[p_chg < 0.98] = 0
		p_chg_98[p_chg >= 0.98] = 1
		cv2.imwrite(os.path.join(out_changemask, '%d.png' % i), 255*p_chg_98)
		print i, float(np.sum(p_chg_98)) / (capture.WIDTH * capture.HEIGHT)		
		pixmodel.update(im)
		i += 1

if __name__ == '__main__':	
	motion_detect_demo(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6])