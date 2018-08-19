import os
import sys
caffe_root = '../mscaffe-crf/'
sys.path.insert(0, caffe_root + 'python')
import caffe
import torch

from utils import upsample_filt, interp_surgery
from PIL import Image
import scipy.io as sio
import numpy as np
import random
import time
import argparse

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
	solver1 = caffe.SGDSolver(args.dss_proto)
	if 'vgg' in args.dss_weights:
		interp_layers = [k for k in solver1.net.params.keys() if 'up' in k]
		interp_surgery(solver1.net, interp_layers)
	solver1.net.copy_from(args.dss_weights)
	print('loaded solver1')

	matfile1 = sio.loadmat(args.datalist1)['trainImgSet']
	datalist1 = [matfile1[i][0][0] for i in range(matfile1.shape[0])]
	matfile2 = sio.loadmat(args.datalist2)['trainImgSet']
	datalist2 = [matfile2[i][0][0] for i in range(matfile2.shape[0])]
	matfile3 = sio.loadmat(args.datalist3)['trainImgSet']
	datalist3 = [matfile3[i][0][0] for i in range(matfile3.shape[0])]

	valinput = './dataset/'+args.valdata+'/imgs/'
	valgt = './dataset/'+args.valdata+'/gt/' 
	valmatfile = './dataset/'+args.valdata+'/valImgSet.mat'
	valmatfile = sio.loadmat(valmatfile)['valImgSet'] 
	vallist = [valmatfile[i][0][0] for i in range(valmatfile.shape[0])] 

	logfile = args.logfile
	if logfile == '': logfile = args.prefix+'.log'
	if os.path.isfile(logfile): os.system('rm '+logfile)

	learn_data3_prob = 0.3
	learn_data2_prob = 0.4
	# learn_data2_prob = 0 
	# learn_data3_prob = 0
	loss_arch = []
	loss_bigarch = []
	start_t = time.time()
	it = args.start_snapshot 
	while it < args.max_iter:

		'''if it < args.max_iter/3 and it+1 >= args.max_iter/3:
			learn_data3_prob = 0.6
			learn_data2_prob = 0.3
		if it < args.max_iter*2/3 and it+1 >= args.max_iter*2/3:
			learn_data3_prob = 0.4
			learn_data2_prob = 0.4'''

		tmpinput = args.inputdir1
		tmpgt = args.gtdir1
		tmplist = datalist1
		tmpext = '.jpg'
		r = random.uniform(0.,1.)
		if r < learn_data3_prob:
			tmpinput = args.inputdir3
			tmpgt = args.gtdir3
			tmplist = datalist3
			tmpext = '.png'
		elif r < learn_data2_prob + learn_data3_prob:
			tmpinput = args.inputdir2 
			tmpgt = args.gtdir2 
			tmplist = datalist2 

		i = it%len(tmplist) 
		gt = Image.open(tmpgt + tmplist[i][:-4] + '.png')
		img = Image.open(tmpinput + tmplist[i][:-4] + tmpext)
		if random.random() > args.flip_prob:
			gt = gt.transpose(Image.FLIP_LEFT_RIGHT)
			img = img.transpose(Image.FLIP_LEFT_RIGHT)
		imgw, imgh = img.size

		gt = preprocess_gt(gt)
		img = vgg_preprocess(img)
		
		solver1.net.clear_param_diffs()
		solver1.net.blobs['data'].reshape(*img.shape)
		solver1.net.blobs['data'].data[...] = img
		solver1.net.blobs['label'].reshape(*gt.shape)
		solver1.net.blobs['label'].data[...] = gt
		solver1.net.forward()

		loss = solver1.net.blobs['loss-fuse'].data.copy()
		if len(loss_arch) < args.display_every:
			loss_arch.append(float(loss))
		else:
			loss_arch[it % args.display_every] = float(loss)
		if len(loss_bigarch) < args.snapshot_every:
			loss_bigarch.append(float(loss)) 
		else:
			loss_bigarch[it % args.snapshot_every] = float(loss)

		# sigmoid_fuse = solver1.net.blobs['sigmoid-fuse'].data.copy()

		solver1.net.backward()
		solver1.apply_update()
		solver1.increment_iter()

		if it % args.display_every == 0:
			meanloss = sum(loss_arch) * 1.0 / len(loss_arch)
			print >> sys.stderr, "[%s] Iteration %d: %.2f seconds loss:%.4f" % (
				time.strftime("%c"), it, time.time() - start_t, meanloss)

		if it % args.snapshot_every == 0:
			trainloss = sum(loss_bigarch) * 1.0 / len(loss_bigarch) 
			vallosses = []
			tmpdir = 'tmp/'
			if os.path.isdir(tmpdir):
				os.system('rm '+tmpdir+'*')
			else:
				os.makedirs(tmpdir) 
			for j in range(len(vallist)):
				gt = Image.open(valgt+vallist[j][:-4]+'.png') 
				img = Image.open(valinput+vallist[j][:-4]+'.jpg') 
				gt = preprocess_gt(gt) 
				img = vgg_preprocess(img)
				solver1.net.clear_param_diffs()
				solver1.net.blobs['data'].reshape(*img.shape)
				solver1.net.blobs['data'].data[...] = img
				solver1.net.blobs['label'].reshape(*gt.shape)
				solver1.net.blobs['label'].data[...] = gt
				solver1.net.forward()
				loss = solver1.net.blobs['loss-fuse'].data.copy()
				vallosses.append(float(loss))
				sigmoid_fuse = solver1.net.blobs['sigmoid-fuse'].data.copy()
				pred = Image.fromarray(np.squeeze(np.rint(sigmoid_fuse*255.0).astype(np.uint8)))
				pred.save(tmpdir+vallist[j][:-4]+'.png')
			valloss = sum(vallosses) * 1.0 / len(vallosses)
			import matlab.engine
			eng  = matlab.engine.start_matlab()
			eng.addpath('/research/adv_saliency/evaluation')
			mae,p,r,fm = eng.callEvalFunc(tmpdir, valgt, nargout=4)
			with open(logfile,'a') as f:
				f.write('iter:%d trainloss:%.4f valloss:%.4f mae:%.4f p:%.4f r:%.4f f:%.4f\n'%(
					it,trainloss,valloss,mae,p,r,fm))

			curr_snapshot_folder = args.snapshot_folder +'/' + str(it)
			print >> sys.stderr, '\n === Saving snapshot to ' + curr_snapshot_folder + ' ===\n'
			solver1.snapshot()
		it = it + 1

def get_arguments():
	parser = argparse.ArgumentParser()
	parser.add_argument('-prefix'           , type=str ,  default= 'dss_v6')
	parser.add_argument('-start_snapshot'   , type=int ,  default= 0)
	parser.add_argument('-max_iter'         , type=int ,  default= 200000)
	parser.add_argument('-display_every'    , type=int ,  default= 10)
	parser.add_argument('-snapshot_every'   , type=int ,  default= 2500)
	parser.add_argument('-snapshot_folder'  , type=str ,  default= 'snapshots/')
	parser.add_argument('-verbose'          , type=int ,  default= 1)
	parser.add_argument('-flip_prob'        , type=float ,default= 0.5)
	# parser.add_argument('-dss_weights'      , type=str ,  default= 'vgg16.caffemodel')
	# parser.add_argument('-dss_weights'      , type=str ,  default= 'dss.caffemodel')
	parser.add_argument('-dss_weights'      , type=str ,  default= './snapshots/dss_v5_iter_10001.caffemodel')
	parser.add_argument('-dss_proto'        , type=str ,  default= 'solver_dss.prototxt')
	parser.add_argument('-inputdir1'        , type=str ,  default= './dataset/MSRA-B/imgs/')
	parser.add_argument('-inputdir2'        , type=str ,  default= './dataset/DUT-OMRON/imgs/')
	parser.add_argument('-inputdir3'        , type=str ,  default= './dataset/HKU-IS/imgs/')
	parser.add_argument('-gtdir1'           , type=str ,  default= './dataset/MSRA-B/gt/')
	parser.add_argument('-gtdir2'           , type=str ,  default= './dataset/DUT-OMRON/gt/')
	parser.add_argument('-gtdir3'           , type=str ,  default= './dataset/HKU-IS/gt/')
	parser.add_argument('-datalist1'        , type=str ,  default= './dataset/MSRA-B/trainImgSet.mat')
	parser.add_argument('-datalist2'        , type=str ,  default= './dataset/DUT-OMRON/trainImgSet.mat') 
	parser.add_argument('-datalist3'        , type=str ,  default= './dataset/HKU-IS/trainImgSet.mat')
	parser.add_argument('-valdata'          , type=str ,  default= 'DUT-OMRON') 
	parser.add_argument('-logfile'          , type=str ,  default= '')
	return parser.parse_args()

if __name__ == '__main__':
	args = get_arguments()
	main(args)