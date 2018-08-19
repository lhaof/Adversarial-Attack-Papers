import os
import sys
caffe_root = '../mscaffe-crf/'
sys.path.insert(0, caffe_root + 'python')
import caffe
import torch

from utils import upsample_filt, interp_surgery, NegProb, np_softmax
from PIL import Image
import pylab as pl
import scipy.io as sio
import numpy as np
import random
import time
import argparse

def get_datalist(dataroot, dataset, split):
	matpath = dataroot + dataset + '/' + split + 'ImgSet.mat' 
	matfile = sio.loadmat(matpath)[split + 'ImgSet']
	datalist = [matfile[i][0][0] for i in range(matfile.shape[0])] 
	return datalist 

def sample_data_and_list(args):
	r = random.uniform(0.,1.)
	tmpdata = args.dataset3 
	tmplist = get_datalist(args.dataroot,args.dataset3,'train')
	ext = '.png'
	if r < args.learn_data1_prob:
		tmpdata = args.dataset1
		tmplist = get_datalist(args.dataroot,args.dataset1,'train')
		ext = '.jpg'
	elif r < args.learn_data1_prob + args.learn_data2_prob: 
		tmpdata = args.dataset2 
		tmplist = get_datalist(args.dataroot,args.dataset2,'train')
		ext = '.jpg'
	return tmpdata, tmplist, ext 

def preprocess_all(gt, img, img2, flip_prob, crfsize):
	if random.random()>flip_prob:
		gt = gt.transpose(Image.FLIP_LEFT_RIGHT)
		img = img.transpose(Image.FLIP_LEFT_RIGHT)
		img2 = img2.transpose(Image.FLIP_LEFT_RIGHT)
	imgw, imgh = img.size
	if imgw>crfsize or imgh>crfsize:
		if imgw>imgh:
			imgh = int(round(imgh*1.0*crfsize/imgw))
			imgw = crfsize
		if imgh>imgw:
			imgw = int(round(imgw*1.0*crfsize/imgh))
			imgh = crfsize		
		img = img.resize((imgw,imgh))
		img2 = img2.resize((imgw,imgh))
		gt = gt.resize((imgw,imgh))
	return gt, img, img2, imgw, imgh 

def preprocess_gt(gt):
	gt = np.expand_dims(np.expand_dims(np.array(gt),axis=0),axis=0)
	gt = gt / max(1e-6,gt.max())
	return gt 

def vgg_preprocess(img):
	img = np.array(img)
	if len(img.shape) == 2:
		img = np.tile(np.expand_dims(img,axis=2),(1,1,3))
	if img.shape[2] == 1:
		img = np.concatenate((img,img,img),axis=2) 
	img = img[:,:,::-1] - np.array((104.00698793,116.66876762,122.67891434))
	img = img.transpose((2,0,1))
	img = np.expand_dims(img, axis=0)
	return img

def main(args):
	caffe.set_mode_gpu()
	caffe.set_device(0)
	solver1 = caffe.SGDSolver(args.net_proto)
	if 'vgg' in args.net_weights:
		interp_layers = [k for k in solver1.net.params.keys() if 'up' in k]
		interp_surgery(solver1.net, interp_layers)
	solver1.net.copy_from(args.net_weights)

	cuda0 = torch.device('cuda:0')
	negprob = NegProb()
	negprob.cuda(0)
	negprob.train()
	solver2 = caffe.SGDSolver(args.crf_proto) 

	crfsize = 500
	input1_ = np.zeros(shape=(1,2,crfsize,crfsize))
	input2_ = np.zeros(shape=(1,3,crfsize,crfsize))
	label_ = np.zeros(shape=(1,1,crfsize,crfsize))
	img = np.ndarray((1,))
	img2 = np.ndarray((1,))
	gt = np.ndarray((1,))

	datalist1 = get_datalist(args.dataroot, args.dataset1, 'train')
	datalist2 = get_datalist(args.dataroot, args.dataset2, 'train') 
	datalist3 = get_datalist(args.dataroot, args.dataset3, 'train') 
	vallist = get_datalist(args.dataroot, args.valdata, args.valsplit) 
	vallist.sort()
	valindir1 = args.dataroot + args.valdata + args.valposfix + '/'
	valindir2 = valindir1 
	valext = '.jpg'
	if 'adv' in args.valposfix: 
		valindir1 += 'round_Linf_25_shuffle-seg3000-10/'
		valindir2 += 'round_Linf_25_sgs3fbf1/'
		valext = '.png'
	else:
		valindir1 += 'imgs_shuffle-seg3000-10/'
		valindir2 += 'imgs_sgs3fbf1/'
	valgtdir = args.dataroot + args.valdata +'/gt/'
	logfile = args.prefix + '.log'
	if os.path.isfile(logfile): os.system('rm '+logfile) 

	loss_bigarch = []
	loss_arch = [] 
	start_t = time.time()
	it = args.start_snapshot
	while it < args.max_iter:
		orgw = -1
		orgh = -1
		imgw = -1
		imgh = -1
		r = random.uniform(0.,1.)
		if r < args.shuffle_prob:
			r = random.uniform(0.,1.)
			if r < args.online_sfl_prob:
				## do online shuffling
				tmpdata, tmplist, ext = sample_data_and_list(args)
				tmpindir1 = args.dataroot + tmpdata + '/imgs/' 
				tmpindir2 = args.dataroot + tmpdata + '/imgs_sgs3fbf1/'
				tmpsegdir = args.dataroot + tmpdata + '/imgs_seg3000/'
				tmpgtdir = args.dataroot + tmpdata + '/gt/'
				i = it % len(tmplist) 
				img = pl.imread(tmpindir1 + tmplist[i][:-4] + ext)
				img.setflags(write=True) 
				if len(img.shape) == 2:
					img = np.expand_dims(img,axis=2)
				if img.shape[2] == 1:
					img = np.concatenate((img,img,img),axis=2)
				seg = sio.loadmat(tmpsegdir + tmplist[i][:-4] + '.mat')['seg']
				segids = np.unique(seg)
				for j in range(segids.shape[0]):
					sid = segids[j]
					h_ind, w_ind = np.where(seg==sid)
					perm = np.random.permutation(h_ind.shape[0])
					seg_rgb = img[h_ind,w_ind,:]
					seg_rgb = seg_rgb[perm,:]
					img[h_ind,w_ind,:] = seg_rgb
				if np.max(img) <= 1: img = img * 255.0
				img = Image.fromarray(np.rint(img).astype(np.uint8),'RGB')
				img2 = Image.open(tmpindir2 + tmplist[i][:-4] + ext) 
				orgw, orgh = img.size
				gt = Image.open(tmpgtdir + tmplist[i][:-4] + '.png')
				gt, img, img2, imgw, imgh = preprocess_all(gt,img,img2,args.flip_prob,args.crfsize)
				gt = preprocess_gt(gt)
				img = vgg_preprocess(img)
				img2 = vgg_preprocess(img2) 	 
			else:
				## do offline shuffling 
				tmpdata, tmplist, ext = sample_data_and_list(args)
				tmpindir1 = args.dataroot + tmpdata + '/imgs_shuffle-seg3000-10/'
				tmpindir2 = args.dataroot + tmpdata + '/imgs_sgs3fbf1/'
				tmpgtdir = args.dataroot + tmpdata + '/gt/'
				i = it % len(tmplist)
				img = Image.open(tmpindir1 + tmplist[i][:-4] + ext)
				img2 = Image.open(tmpindir2 + tmplist[i][:-4] + ext) 
				orgw, orgh = img.size
				gt = Image.open(tmpgtdir + tmplist[i][:-4] + '.png') 
				gt, img, img2, imgw, imgh = preprocess_all(gt,img,img2,args.flip_prob,args.crfsize)
				gt = preprocess_gt(gt)
				img = vgg_preprocess(img)
				img2 = vgg_preprocess(img2) 
		else:
			tmpdata, tmplist, ext = sample_data_and_list(args)
			tmpindir1 = args.dataroot + tmpdata + '/imgs/' 
			tmpindir2 = tmpindir1
			tmpgtdir = args.dataroot + tmpdata + '/gt/' 
			i = it % len(tmplist)
			img = Image.open(tmpindir1 + tmplist[i][:-4] + ext)
			img2 = Image.open(tmpindir2 + tmplist[i][:-4] + ext) 
			orgw, orgh = img.size 
			gt = Image.open(tmpgtdir + tmplist[i][:-4] + '.png') 
			gt, img, img2, imgw, imgh = preprocess_all(gt,img,img2,args.flip_prob,args.crfsize)
			gt = preprocess_gt(gt)
			img = vgg_preprocess(img)
			img2 = vgg_preprocess(img2) 

		solver1.net.clear_param_diffs()
		solver1.net.blobs['data'].reshape(*img.shape)
		solver1.net.blobs['data'].data[...] = img
		solver1.net.blobs['label'].reshape(*gt.shape)
		solver1.net.blobs['label'].data[...] = gt
		solver1.net.forward()
		sigmoid_fuse = solver1.net.blobs['sigmoid-fuse'].data.copy()
		input1_[:,0,:,:] = 1
		input1_[:,1,:,:] = 0
		input1_[:,0,:imgh,:imgw] = sigmoid_fuse * (-1.0) + 1.0
		input1_[:,1,:imgh,:imgw] = sigmoid_fuse
		input1_t = torch.tensor(input1_, dtype=torch.float32, device=cuda0,
			requires_grad=True)
		with torch.enable_grad():
			neg_prob_t = torch.log( torch.clamp(input1_t, 1e-6, 1.0) )
		neg_prob_arr = neg_prob_t.data.cpu().numpy()

		input2_[:,:,:,:] = 0
		input2_[:,:,:imgh,:imgw] = img2
		label_[:,:,:,:] = 0
		label_[:,:,:imgh,:imgw] = gt

		solver2.net.clear_param_diffs()
		solver2.net.blobs['coarse'].data[...] = neg_prob_arr
		solver2.net.blobs['data'].data[...] = input2_
		solver2.net.blobs['label'].data[...] = label_
		solver2.net.forward()
		solver2.net.backward()
		coarse_diff = solver2.net.blobs['coarse'].diff.copy()
		data_diff = solver2.net.blobs['data'].diff.copy()
		solver2.apply_update()
		solver2.increment_iter()
		loss = solver2.net.blobs['loss'].data.copy()
		if len(loss_arch) < args.display_every:
			loss_arch.append(float(loss))
		else:
			loss_arch[it % args.display_every] = float(loss)
		if len(loss_bigarch) < args.snapshot_every: 
			loss_bigarch.append(float(loss)) 
		else:
			loss_bigarch[it % args.snapshot_every] = float(loss) 
		neg_prob_diff = solver2.net.blobs['coarse'].diff.copy()
		neg_prob_diff_t = torch.tensor(neg_prob_diff).to(cuda0)
		neg_prob_t.backward(neg_prob_diff_t)
		sigmoid_fuse_diff = input1_t.grad.cpu().numpy()
		solver1.net.blobs['sigmoid-fuse'].diff[...] = sigmoid_fuse_diff[:,1,:imgh,:imgw] - sigmoid_fuse_diff[:,0,:imgh,:imgw]
		solver1.net.backward()
		solver1.apply_update()
		solver1.increment_iter()

		if it % args.display_every == 0:
			meanloss = sum(loss_arch)*1.0/len(loss_arch)
			print >> sys.stderr, "[%s] Iteration %d: %.2f seconds loss:%.4f" % (
				time.strftime("%c"), it, time.time()-start_t, meanloss)
		if it % args.snapshot_every == 0 and it >= 0:
			## do validation here 
			print >> sys.stderr, 'iter:%d doing validation...'%(it)

			tmpdir1 = 'tmp1/'
			tmpdir2 = 'tmp2/'
			if os.path.isdir(tmpdir1):
				os.system('rm '+tmpdir1+'*')
			else:
				os.makedirs(tmpdir1)
			if os.path.isdir(tmpdir2):
				os.system('rm '+tmpdir2+'*')
			else:
				os.makedirs(tmpdir2) 
			vallosses = []
			valstart_t = time.time()
			for j in range(len(vallist)):
				img = Image.open(valindir1 + vallist[j][:-4] + valext) 
				img2 = Image.open(valindir2 + vallist[j][:-4] + valext) 
				orgw, orgh = img.size
				gt = Image.open(valgtdir + vallist[j][:-4] + '.png') 
				gt,img,img2,imgw,imgh = preprocess_all(gt,img,img2,0,args.crfsize)
				gt = preprocess_gt(gt) 
				img = vgg_preprocess(img) 
				img2 = vgg_preprocess(img2) 
				solver1.net.clear_param_diffs()
				solver1.net.blobs['data'].reshape(*img.shape)
				solver1.net.blobs['data'].data[...] = img
				solver1.net.blobs['label'].reshape(*gt.shape)
				solver1.net.blobs['label'].data[...] = gt
				solver1.net.forward()
				sigmoid_fuse = solver1.net.blobs['sigmoid-fuse'].data.copy()
				pred1 = Image.fromarray(np.squeeze(np.rint(sigmoid_fuse*255.0).astype(np.uint8)))
				pred1.save(tmpdir1 + vallist[j][:-4] + '.png') 
				input1_[:,0,:,:] = 1
				input1_[:,1,:,:] = 0
				input1_[:,0,:imgh,:imgw] = sigmoid_fuse * (-1.0) + 1.0
				input1_[:,1,:imgh,:imgw] = sigmoid_fuse
				input1_t = torch.tensor(input1_, dtype=torch.float32, device=cuda0,
					requires_grad=True)
				with torch.enable_grad():
					neg_prob_t = torch.log( torch.clamp(input1_t, 1e-6, 1.0) )
				neg_prob_arr = neg_prob_t.data.cpu().numpy()
				input2_[:,:,:,:] = 0
				input2_[:,:,:imgh,:imgw] = img2
				label_[:,:,:,:] = 0
				label_[:,:,:imgh,:imgw] = gt
				solver2.net.clear_param_diffs()
				solver2.net.blobs['coarse'].data[...] = neg_prob_arr
				solver2.net.blobs['data'].data[...] = input2_
				solver2.net.blobs['label'].data[...] = label_
				solver2.net.forward()
				loss = solver2.net.blobs['loss'].data.copy()
				vallosses.append(float(loss)) 
				pred = solver2.net.blobs['softmax'].data.copy()
				pred2 = pred[:,1,:imgh,:imgw]
				pred2 = (pred2 - pred2.min()) / max(1e-6, pred2.max()-pred2.min()) 
				pred2 = Image.fromarray(np.squeeze(np.rint(pred2*255.0).astype(np.uint8)))
				pred2.save(tmpdir2 + vallist[j][:-4] + '.png')
				if j % 100 == 0:
					print >> sys.stderr, 'validating sample %d time-cost %.2f secs'%(j,time.time()-valstart_t)

			valloss = sum(vallosses) * 1.0 / len(vallosses)
			trainloss = sum(loss_bigarch) * 1.0 / len(loss_bigarch) 
			import matlab.engine
			eng  = matlab.engine.start_matlab()
			eng.addpath('/research/adv_saliency/evaluation')
			mae1,p1,r1,fm1 = eng.callEvalFunc(tmpdir1, valgtdir, nargout=4)
			mae2,p2,r2,fm2 = eng.callEvalFunc(tmpdir2, valgtdir, nargout=4)
			with open(logfile,'a') as f:
				f.write( 
					'iter:%d trainloss:%.4f valloss:%.4f '%(it,trainloss,valloss) + 
					'mae1:%.4f p1:%.4f r1:%.4f f1:%.4f '%(mae1,p1,r1,fm1) + 
					'mae2:%.4f p2:%.4f r2:%.4f f2:%.4f\n'%(mae2,p2,r2,fm2)
				)

			curr_snapshot_folder = args.snapshot_folder +'/' + str(it)
			print >> sys.stderr, '\n === Saving snapshot to ' + curr_snapshot_folder + ' ===\n'
			solver1.snapshot()
			solver2.snapshot()
		it = it+1

def get_arguments():
	parser = argparse.ArgumentParser()
	parser.add_argument('-prefix'         , type=str , default= 'rosa_v5')
	parser.add_argument('-start_snapshot' , type=int , default= 0) 
	parser.add_argument('-max_iter'       , type=int , default= 1000000000)
	parser.add_argument('-display_every'  , type=str , default= 10) 
	parser.add_argument('-snapshot_every' , type=int , default= 500) 
	parser.add_argument('-snapshot_folder', type=str , default= 'snapshots/')
	parser.add_argument('-flip_prob'      , type=float,default= 0.5) 
	parser.add_argument('-crfsize'        , type=int , default= 500)
	parser.add_argument('-crf_proto'      , type=str , default= 'solver_crf.prototxt')
	parser.add_argument('-net_proto'      , type=str , default= 'solver_dss.prototxt')
	parser.add_argument('-net_weights'    , type=str , default= './snapshots/dss_v5_iter_10001.caffemodel')
	parser.add_argument('-dataroot'       , type=str , default= './dataset/')
	parser.add_argument('-dataset1'       , type=str , default= 'DUT-OMRON')
	parser.add_argument('-dataset2'       , type=str , default= 'MSRA-B') 
	parser.add_argument('-dataset3'       , type=str , default= 'HKU-IS')
	parser.add_argument('-valdata'        , type=str , default= 'DUT-OMRON')
	parser.add_argument('-valposfix'      , type=str , default= '-adv-v2')
	parser.add_argument('-valsplit'       , type=str , default= 'val') 
	parser.add_argument('-online_sfl_prob', type=float,default= 0.0)
	parser.add_argument('-shuffle_prob'   , type=float,default= 1.0) 
	parser.add_argument('-learn_data1_prob',type=float,default= 0.6) 
	parser.add_argument('-learn_data2_prob',type=float,default= 0.2) 
	parser.add_argument('-learn_data3_prob',type=float,default= 0.2) 
	return parser.parse_args()

if __name__ == '__main__':
	args = get_arguments()
	main(args)