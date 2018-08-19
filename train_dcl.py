import os
import sys
caffe_root = '../mscaffe-crf/'
sys.path.insert(0, caffe_root + 'python')
import caffe
import torch
import torch.nn.functional as F

from utils import NegProb, np_softmax
from PIL import Image
import scipy.io as sio
import numpy as np
import random
import time

prefix = 'dcl_'
verbose = True
flip_prob = 0.5
inputsize = 513
start_snapshot = 0
max_iter = 1000000000
display_every = 10
snapshot_every = 2500
snapshot_folder = 'snapshots_dcl'
if not os.path.exists(snapshot_folder):
	os.makedirs(snapshot_folder) 
snapshot_at_iter_list = [2500]

caffe.set_mode_gpu()
caffe.set_device(0)
cuda0 = torch.device('cuda:0')

solver1 = caffe.SGDSolver('solver_dcl_train.prototxt')
solver1_weights = './exper_saliency/model/DCL_Saliency.caffemodel'
solver1.net.copy_from(solver1_weights)
print('loaded solver1')

gt_ = np.zeros(shape=(1,1,inputsize,inputsize))
img_ = np.zeros(shape=(1,3,inputsize,inputsize))
weight_ = np.zeros(shape=(1,1,inputsize,inputsize))

inputdir1 = './dataset/MSRA-B/imgs_shuffle-seg3000-10/'
inputdir2 = './dataset/MSRA-B/imgs_sgs3fbf1/'
gtdir = './dataset/MSRA-B/gt/'
datalist = './dataset/MSRA-B/trainImgSet.mat'
matfile = sio.loadmat(datalist)
matfile = matfile['trainImgSet']
datalist = [matfile[i][0][0] for i in range(matfile.shape[0])]

start_t = time.time()
loss_arch = np.zeros(shape=(snapshot_every,),dtype=np.float32)
it = start_snapshot
while it<max_iter:
	if it%snapshot_every==0 or it in snapshot_at_iter_list:
		verbose = True
	else:
		verbose = False
	i = it%len(datalist)
	gtname = gtdir + datalist[i][:-4]+'.png'
	gt = Image.open(gtname)
	imgname = inputdir1 + datalist[i][:-4]+'.jpg'
	imgname2 = inputdir2 + datalist[i][:-4]+'.jpg'
	img = Image.open(imgname)
	img2 = Image.open(imgname2)
	if verbose:
		gt.save(prefix+str(it)+'_gt.png')
		img.save(prefix+str(it)+'_img.png')
		img2.save(prefix+str(it)+'_img2.png')
	if random.random()>flip_prob:
		gt = gt.transpose(Image.FLIP_LEFT_RIGHT)
		img = img.transpose(Image.FLIP_LEFT_RIGHT)
		img2 = img2.transpose(Image.FLIP_LEFT_RIGHT)
	orgw, orgh = img.size
	imgw, imgh = img.size
	if imgw>inputsize or imgh>inputsize:
		if imgw>imgh:
			imgh = int(round(imgh*1.0*inputsize/imgw))
			imgw = inputsize
		if imgh>imgw:
			imgw = int(round(imgw*1.0*inputsize/imgh))
			imgh = inputsize		
		img = img.resize((imgw,imgh))
		img2 = img2.resize((imgw,imgh))
		gt = gt.resize((imgw,imgh))

	gt = np.array(gt)
	gt = np.expand_dims(gt, axis=0)
	gt = np.expand_dims(gt, axis=0)
	gt = gt / max(1e-6,gt.max())
	gt_[:,:,:,:] = 0
	gt_[:,:,:imgh,:imgw] = gt
	img = np.array(img)
	img2 = np.array(img2)
	if len(img.shape)==2:
		img = np.expand_dims(img, axis=2)
		img = np.tile(img, (1,1,3))
		img2 = np.expand_dims(img2, axis=2)
		img2 = np.tile(img2, (1,1,3))
	img = img[:,:,::-1] - np.array((104.00698793,116.66876762,122.67891434))
	img = img.transpose((2,0,1))
	img = np.expand_dims(img, axis=0)
	img2 = img2[:,:,::-1] - np.array((104.00698793,116.66876762,122.67891434))
	img2 = img2.transpose((2,0,1))
	img2 = np.expand_dims(img2, axis=0)
	img_[:,:,:,:] = 0
	img_[:,:,:imgh,:imgw] = img
	weight_[:,:,:,:] = 0
	weight_[:,:,:imgh,:imgw] = 1
	solver1.net.clear_param_diffs()
	solver1.net.blobs['data'].data[...] = img_
	solver1.net.forward()
	sm = solver1.net.blobs['fc8_saliency_reg'].data.copy()
	gt_t = torch.tensor(gt_, dtype=torch.float32, device=cuda0, requires_grad=False)
	sm_t = torch.tensor(sm, dtype=torch.float32, device=cuda0, requires_grad=True)
	weight_t = torch.tensor(weight_, dtype=torch.float32, device=cuda0, requires_grad=False)
	loss = F.binary_cross_entropy(sm_t, gt_t, weight=weight_t, size_average=False)
	if verbose:
		pred1 = Image.fromarray(np.squeeze(np.rint(sm[:,:,:imgh,:imgw] * 255.0).astype(np.uint8)))
		if orgw!=imgw or orgh!=imgh:
			pred1 = pred1.resize((orgw,orgh))
		pred1.save(prefix+str(it)+'_pred1.png')
	
	loss_arch[it%snapshot_every] = float(loss)
	loss.backward()
	solver1.net.blobs['fc8_saliency_reg'].diff[...] = sm_t.grad.cpu().numpy()
	solver1.net.backward()
	solver1.apply_update()
	solver1.increment_iter()
	
	if it%display_every==0:
		meanloss = 0
		cnt1 = it % snapshot_every + 1
		if cnt1 >= display_every:
			meanloss = loss_arch[cnt1 - display_every + 1:cnt1].mean()
		elif it < snapshot_every:
			meanloss = loss_arch[:cnt1].mean()
		else:
			cnt2 = display_every - cnt1
			meanloss = ( loss_arch[:cnt1].sum() + loss_arch[snapshot_every - cnt2:].sum() )/display_every
		print >> sys.stderr, "[%s] Iteration %d: %.2f seconds loss:%.4f" % (
			time.strftime("%c"), it, time.time()-start_t, meanloss)
	if it%snapshot_every==0 or it in snapshot_at_iter_list:
		curr_snapshot_folder = snapshot_folder +'/' + str(it)
		print >> sys.stderr, '\n === Saving snapshot to ' + curr_snapshot_folder + ' ===\n'
		solver1.snapshot()
		if it >= snapshot_every:
			trainloss = loss_arch.mean()
			print >> sys.stderr, "\n latest train loss: %.4f " % (trainloss), "\n"
	it = it+1
	#break
