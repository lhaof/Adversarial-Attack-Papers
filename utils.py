import random
import numpy as np
import scipy.io as sio
import os
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

def np_softmax(X, theta = 1.0, axis = None):
	"""
	Compute the softmax of each element along an axis of X.

	Parameters
	----------
	X: ND-Array. Probably should be floats. 
	theta (optional): float parameter, used as a multiplier
		prior to exponentiation. Default = 1.0
	axis (optional): axis to compute values along. Default is the 
		first non-singleton axis.

	Returns an array the same size as X. The result will sum to 1
	along the specified axis.
	"""

	# make X at least 2d
	y = np.atleast_2d(X)

	# find axis
	if axis is None:
		axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

	# multiply y against the theta parameter, 
	y = y * float(theta)

	# subtract the max for numerical stability
	y = y - np.expand_dims(np.max(y, axis = axis), axis)

	# exponentiate y
	y = np.exp(y)

	# take the sum along the specified axis
	ax_sum = np.expand_dims(np.sum(y, axis = axis), axis)

	# finally: divide elementwise
	p = y / ax_sum

	# flatten if X was 1D
	if len(X.shape) == 1: p = p.flatten()

	return p

class NegProb(nn.Module):
	def __init__(self):
		super(NegProb, self).__init__()

	def forward(self, x, y):
		return torch.mul( torch.log( torch.max(x, y) ), -1.0 )

def prepro_image(img):
	if len(img.shape)==2:
		img = np.expand_dims(img,axis=2)
		img = np.concatenate((img,img,img),axis=2)
	img = img[:,:,::-1] - np.array((104.00698793,116.66876762,122.67891434))
	img = np.transpose(img,(2,0,1))
	img = np.expand_dims(img,axis=0)
	return img

def prepro_label(label):
	label = label / np.max(label)
	label = np.expand_dims(label,axis=0)
	label = np.expand_dims(label,axis=0)
	return label

class Dataloader:
	def __init__(self, 
		inputdir1, inputdir2, labeldir, datalist,
		inputext, labelext, prefix):
		self.inputdir1 = inputdir1
		self.inputdir2 = inputdir2
		self.labeldir = labeldir
		self.inputext = inputext
		self.labelext = labelext
		self.prefix = prefix
		self.imgset = os.path.basename(datalist).split('.')[0]
		matfile = sio.loadmat(datalist)
		matfile = matfile[self.imgset]
		self.datalist = [matfile[i][0][0] for i in range(matfile.shape[0])]
		self.ptr = 0

	def shuffle(self):
		random.shuffle(self.datalist)

	def getBatch(self, 
		batch_size, flip_prob, max_size,
		verbose, return_name, iters):
		input1 = []
		input2 = []
		label = []
		namelist = []
		imgh = 0
		imgw = 0
		imgslist = []
		for i in range(batch_size):
			ix = (self.ptr+i) % len(self.datalist)
			names = [
				self.inputdir1+self.datalist[ix][:-4]+self.inputext,
				self.inputdir2+self.datalist[ix][:-4]+self.inputext,
				self.labeldir+self.datalist[ix][:-4]+self.labelext
			]
			if return_name:
				namelist.append(self.datalist[ix][:-4]+self.labelext)
			imgs = [Image.open(x) for x in names]
			if verbose:
				imgs[0].save(self.prefix+str(iters)+'_input1.png')
				imgs[1].save(self.prefix+str(iters)+'_input2.png')
				imgs[2].save(self.prefix+str(iters)+'_label.png')
			if random.uniform(0,1) > flip_prob:
				imgs = [x.transpose(Image.FLIP_LEFT_RIGHT) for x in imgs]
			imgslist.append(imgs)

		imgw, imgh = imgslist[0][0].size
		if imgw > imgh and imgw > max_size:
			imgh = int(round(imgh*1.0*max_size/imgw))
			imgw = max_size
		if imgh > imgw and imgh > max_size:
			imgw = int(round(imgw*1.0*max_size/imgh))
			imgh = max_size
		for i in range(batch_size):
			imgslist[i] = [np.array(img.resize((imgw,imgh)), dtype=np.float32) for img in imgslist[i]]
			input1.append(prepro_image(imgslist[i][0]))
			input2.append(prepro_image(imgslist[i][1]))
			label.append(prepro_label(imgslist[i][2]))

		self.ptr = (self.ptr+batch_size) % len(self.datalist)
		input1 = np.concatenate(input1,axis=0)
		input2 = np.concatenate(input2,axis=0)
		label = np.concatenate(label, axis=0)
		if return_name:
			return input1, input2, label, namelist
		else:
			return input1, input2, label

def upsample_filt(size):
	factor = (size + 1) // 2
	if size % 2 == 1:
		center = factor - 1
	else:
		center = factor - 0.5
	og = np.ogrid[:size, :size]
	return (1 - abs(og[0] - center) / factor) * \
		(1 - abs(og[1] - center) / factor)

# set parameters s.t. deconvolutional layers compute bilinear interpolation
# N.B. this is for deconvolution without groups
def interp_surgery(net, layers):
	for l in layers:
		m, k, h, w = net.params[l][0].data.shape
		if m != k:
			print 'input + output channels need to be the same'
			raise
		if h != w:
			print 'filters need to be square'
			raise
		filt = upsample_filt(h)
		net.params[l][0].data[range(m), range(k), :, :] = filt
