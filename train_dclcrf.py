import os
import sys
caffe_root = '../mscaffe-crf/'
sys.path.insert(0, caffe_root + 'python')
import caffe
import torch

from utils import NegProb, np_softmax
from PIL import Image
import scipy.io as sio
import numpy as np
import random
import time
import progressbar
bar = progressbar.ProgressBar()

USE_WEIGHTS_CNN = './snapshots_dclcrf-v3/dcl_iter_87501.caffemodel'
USE_WEIGHTS_CRF = './snapshots_dclcrf-v3/crf_iter_87501.caffemodel'
USE_DATA = 'msrab'
USE_SPLIT = 'train'
base_lr_1 = 1e-12
base_lr_2 = 1e-12
prefix = 'dclcrf-v4'
logfile = prefix+'.log'
with open(logfile,'w') as f:
	f.write('training '+prefix+' with lr1='+str(base_lr_1)+' lr2='+str(base_lr_2)+'\n')

verbose = True
flip_prob = 0.5
inputsize = 513

start_test = 2500
start_snapshot = 0
max_iter = 1000000000
test_every = 2500
display_every = 50
snapshot_every = 2500
snapshot_folder = 'snapshots_'+prefix
if not os.path.exists(snapshot_folder):
	os.makedirs(snapshot_folder)
else:
	os.system('rm '+snapshot_folder+'/*')
snapshot_at_iter_list = [2500]

def write_solver(solver_file, net, base_lr, snapshot_prefix):
	with open(solver_file, 'w') as f:
		f.write('net: \"'+net+'\"\n')
		f.write('base_lr: '+str(base_lr)+'\n')
		f.write('lr_policy: \"fixed\"\n')
		f.write('display: 100\n')
		f.write('max_iter: 1000000000\n')
		f.write('momentum: 0.99\n')
		f.write('weight_decay: 0.0005\n')
		f.write('snapshot: 2500\n')
		f.write('snapshot_prefix: \"'+snapshot_prefix+'\"\n')
		f.write('solver_mode: GPU\n')
		
write_solver('solver_dcl.prototxt', 'deploy_DCL_inference.prototxt',
	base_lr_1, snapshot_folder+'/dcl')
write_solver('solver_crf.prototxt', 'crf.prototxt',
	base_lr_2, snapshot_folder+'/crf')
	
caffe.set_mode_gpu()
caffe.set_device(0)

solver1 = caffe.SGDSolver('solver_dcl.prototxt')
solver1_weights = './exper_saliency/model/DCL_Saliency.caffemodel'
if USE_WEIGHTS_CNN != '': 
	solver1_weights = USE_WEIGHTS_CNN
solver1.net.copy_from(solver1_weights)
print('loaded solver1 with %s'%(solver1_weights))

cuda1 = torch.device('cuda:0')
negprob = NegProb()
negprob.cuda(0)
negprob.train()

solver2 = caffe.SGDSolver('solver_crf.prototxt')
if USE_WEIGHTS_CRF != '':
	solver2.net.copy_from(USE_WEIGHTS_CRF)
print('loaded solver2 %s'%(USE_WEIGHTS_CRF))

crfsize = 500
input1_ = np.zeros(shape=(1,2,crfsize,crfsize))
input2_ = np.zeros(shape=(1,3,crfsize,crfsize))
label_ = np.zeros(shape=(1,1,crfsize,crfsize))
gt_ = np.zeros(shape=(1,1,inputsize,inputsize))
img_ = np.zeros(shape=(1,3,inputsize,inputsize))
weight_ = np.zeros(shape=(1,1,inputsize,inputsize))
sm_diff_ = np.zeros(shape=(1,1,inputsize,inputsize))

splits = ['train','test','val']
msrab_dir1 = './dataset/MSRA-B/imgs_shuffle-seg3000-10/'
msrab_dir2 = './dataset/MSRA-B/imgs_sgs3fbf1/'
msrab_gtdir = './dataset/MSRA-B/gt/'
msrab_root = './dataset/MSRA-B/'
msrab_datalist = {}
for sp in splits:
	matfile = sio.loadmat(msrab_root+sp+'ImgSet.mat')
	matfile = matfile[sp+'ImgSet']
	msrab_datalist[sp] = [matfile[i][0][0] for i in range(matfile.shape[0])]
	msrab_datalist[sp].sort()

#testdir1 = './dataset/HKU-IS/imgs_shuffle-seg3000-10/'
#testdir2 = './dataset/HKU-IS/imgs_sgs3fbf1/'
hkuis_dir1 = './dataset/advDSS/advdata/round_Linf_20_shuffle-seg3000-10/'
hkuis_dir2 = './dataset/advDSS/advdata/round_Linf_20_sgs3fbf1/'
hkuis_gtdir = './dataset/HKU-IS/gt/'
hkuis_root = './dataset/HKU-IS/'
hkuis_datalist = {}
for sp in splits:
	matfile = sio.loadmat(hkuis_root+sp+'ImgSet.mat')
	matfile = matfile[sp+'ImgSet']
	hkuis_datalist[sp] = [matfile[i][0][0] for i in range(matfile.shape[0])]


def load_data(dataset, split, index):
	if dataset == 'msrab':	
		gtname = msrab_gtdir + msrab_datalist[split][index][:-4]+'.png'
		gt = Image.open(gtname)
		imgname = msrab_dir1 + msrab_datalist[split][index][:-4]+'.jpg'
		imgname2 = msrab_dir2 + msrab_datalist[split][index][:-4]+'.jpg'
		img = Image.open(imgname)
		img2 = Image.open(imgname2)
		return gt,img,img2
	if dataset == 'hkuis':	
		dir1 = hkuis_dir1
		dir2 = hkuis_dir2
		if split == 'train':
			dir1 = './dataset/HKU-IS/imgs_shuffle-seg3000-10/'
			dir2 = './dataset/HKU-IS/imgs_sgs3fbf1/'
		gtname = hkuis_gtdir + hkuis_datalist[split][index][:-4]+'.png'
		gt = Image.open(gtname)
		imgname = dir1 + hkuis_datalist[split][index][:-4]+'.png'
		imgname2 = dir2 + hkuis_datalist[split][index][:-4]+'.png'
		img = Image.open(imgname)
		img2 = Image.open(imgname2)
		return gt,img,img2
	

def prepro_data(gt, img, img2):
	imgw,imgh = img.size
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

	gt = np.array(gt)
	gt = np.expand_dims(gt, axis=0)
	gt = np.expand_dims(gt, axis=0)
	gt = gt / max(1e-6,gt.max())
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
	return gt,img,img2


start_t = time.time()
loss_arch = np.zeros(shape=(snapshot_every,),dtype=np.float32)
testloss_arch = np.zeros(shape=(len(hkuis_datalist['test']),),dtype=np.float32)

if not os.path.exists('tmp'):
	os.makedirs('tmp')
else:
	os.system('rm tmp/'+prefix+'*')

it = start_snapshot
while it<max_iter:
	if it%snapshot_every==0 or it in snapshot_at_iter_list:
		verbose = True
	else:
		verbose = False
	
	i = ( it%len(msrab_datalist['train']) if (USE_DATA=='msrab') else it%len(hkuis_datalist['train']) )
	gt,img,img2 = ( load_data('msrab','train',i) if (USE_DATA=='msrab') else load_data('hkuis','train',i) )
	
	if verbose:
		gt.save('tmp/'+prefix+'_'+str(it)+'_gt.png')
		img.save('tmp/'+prefix+'_'+str(it)+'_img.png')
		img2.save('tmp/'+prefix+'_'+str(it)+'_img2.png')
	if random.random()>flip_prob:
		gt = gt.transpose(Image.FLIP_LEFT_RIGHT)
		img = img.transpose(Image.FLIP_LEFT_RIGHT)
		img2 = img2.transpose(Image.FLIP_LEFT_RIGHT)
	orgw, orgh = img.size
	gt,img,img2 = prepro_data(gt,img,img2)
	imgh = img.shape[2]
	imgw = img.shape[3]
	
	img_[:,:,:,:] = 0
	img_[:,:,:imgh,:imgw] = img
	solver1.net.clear_param_diffs()
	solver1.net.blobs['data'].data[...] = img_
	solver1.net.forward()
	sm = solver1.net.blobs['fc8_saliency_reg'].data.copy()
	if verbose:
		pred1 = Image.fromarray(np.squeeze(np.rint(sm[:,:,:imgh,:imgw] * 255.0).astype(np.uint8)))
		if orgw!=imgw or orgh!=imgh:
			pred1 = pred1.resize((orgw,orgh))
		pred1.save('tmp/'+prefix+'_'+str(it)+'_pred1.png')
	input1_[:,0,:,:] = 1
	input1_[:,1,:,:] = 0
	input1_[:,0,:imgh,:imgw] = sm[:,0,:imgh,:imgw] * (-1.0) + 1.0
	input1_[:,1,:imgh,:imgw] = sm[:,0,:imgh,:imgw]
	input1_t = torch.tensor(input1_, dtype=torch.float32, device=cuda1,
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
	if verbose:
		pred = solver2.net.blobs['pred'].data.copy()
		pred = np_softmax(pred,axis=1)
		pred2 = Image.fromarray(np.squeeze(np.rint(pred[0,1,:imgh,:imgw]*255.0).astype(np.uint8)))
		if orgw!=imgw or orgh!=imgh:
			pred2 = pred2.resize((orgw,orgh))
		pred2.save('tmp/'+prefix+'_'+str(it)+'_pred2.png')
		
	loss = solver2.net.blobs['loss'].data.copy()
	loss_arch[it % snapshot_every] = float(loss)
	
	neg_prob_diff = solver2.net.blobs['coarse'].diff.copy()
	neg_prob_diff_t = torch.tensor(neg_prob_diff).to(cuda1)
	neg_prob_t.backward(neg_prob_diff_t)
	sm_diff = input1_t.grad.cpu().numpy()
	sm_diff_[:,:,:,:] = 0
	sm_diff_[:,:,:imgh,:imgw] = sm_diff[:,1,:imgh,:imgw] - sm_diff[:,0,:imgh,:imgw]
	solver1.net.blobs['fc8_saliency_reg'].diff[...] = sm_diff_
	solver1.net.backward()
	solver1.apply_update()
	solver1.increment_iter()
	
	if it%display_every==0:
		meanloss = 0
		cnt1 = it % snapshot_every + 1
		if cnt1 >= display_every:
			meanloss = loss_arch[cnt1 - display_every:cnt1].mean()
		elif it < snapshot_every:
			meanloss = loss_arch[:cnt1].mean()
		else:
			cnt2 = display_every - cnt1
			meanloss = ( loss_arch[:cnt1].sum() + loss_arch[snapshot_every - cnt2:].sum() )/display_every
		print >> sys.stderr, "[%s] Iteration %d: %.2f seconds loss:%.4f" % (
			time.strftime("%c"), it, time.time()-start_t, meanloss)
	
	trainloss = -1
	testloss = -1
	
	if it%snapshot_every==0 or it in snapshot_at_iter_list:
		curr_snapshot_folder = snapshot_folder +'/' + str(it)
		print >> sys.stderr, '\n === Saving snapshot to ' + curr_snapshot_folder + ' ===\n'
		solver1.snapshot()
		solver2.snapshot()
		if it >= snapshot_every:
			trainloss = loss_arch.mean()
			print >> sys.stderr, "\n iter: %d train loss: %.4f" % (it,trainloss), "\n"
	
	
	if it%test_every==0 and it>=start_test:
		print >> sys.stderr, "\n begin testing... \n"
		testloss_list = []
		tmp_savedir1 = 'tmp_savedir1/'
		tmp_savedir2 = 'tmp_savedir2/'
		if not os.path.exists(tmp_savedir1): os.makedirs(tmp_savedir1)
		else: os.system('rm '+tmp_savedir1+'/*')
		if not os.path.exists(tmp_savedir2): os.makedirs(tmp_savedir2)
		else: os.system('rm '+tmp_savedir2+'/*')
		for i in range(len(hkuis_datalist['test'])):
			name = hkuis_datalist['test'][i]
			gt,img,img2 = load_data('hkuis','test',i)
			orgw,orgh = img.size
			gt,img,img2 = prepro_data(gt,img,img2)
			imgh = img.shape[2]
			imgw = img.shape[3]
			img_[:,:,:,:] = 0
			img_[:,:,:imgh,:imgw] = img
			
			solver1.net.blobs['data'].data[...] = img_
			solver1.net.forward()
			
			sm = solver1.net.blobs['fc8_saliency_reg'].data.copy()
			pred1 = Image.fromarray(np.squeeze(np.rint(sm[:,:,:imgh,:imgw] * 255.0).astype(np.uint8)))
			if orgw!=imgw or orgh!=imgh:
				pred1 = pred1.resize((orgw,orgh))
			pred1.save(tmp_savedir1+name[:-4]+'.png')
			
			input1_[:,0,:,:] = 1
			input1_[:,1,:,:] = 0
			input1_[:,0,:imgh,:imgw] = sm[:,0,:imgh,:imgw] * (-1.0) + 1.0
			input1_[:,1,:imgh,:imgw] = sm[:,0,:imgh,:imgw]
			input1_t = torch.tensor(input1_, dtype=torch.float32, device=cuda1,
				requires_grad=True)
			with torch.enable_grad():
				neg_prob_t = torch.log( torch.clamp(input1_t, 1e-6, 1.0) )
			neg_prob_arr = neg_prob_t.data.cpu().numpy()

			input2_[:,:,:,:] = 0
			input2_[:,:,:imgh,:imgw] = img2
			label_[:,:,:,:] = 0
			label_[:,:,:imgh,:imgw] = gt
	
			solver2.net.blobs['coarse'].data[...] = neg_prob_arr
			solver2.net.blobs['data'].data[...] = input2_
			solver2.net.blobs['label'].data[...] = label_
			solver2.net.forward()
			
			pred = solver2.net.blobs['pred'].data.copy()
			pred = np_softmax(pred,axis=1)
			pred2 = Image.fromarray(np.squeeze(np.rint(pred[0,1,:imgh,:imgw]*255.0).astype(np.uint8)))
			if orgw!=imgw or orgh!=imgh:
				pred2 = pred2.resize((orgw,orgh))
			pred2.save(tmp_savedir2+name[:-4]+'.png')
			
			loss = solver2.net.blobs['loss'].data.copy()
			testloss_list.append(float(loss))
			testloss_arch[i] = float(loss)
			
		testloss = sum(testloss_list)/len(testloss_list)
		testloss_ = testloss_arch.mean()
		print >> sys.stderr, "\n iter: %d %d samples testloss: %.4f testloss_: %.4f" % (it,len(testloss_list),testloss,testloss_), "\n"
		os.system('python evaluate.py '+tmp_savedir1+' '+hkuis_gtdir)
		os.system('python evaluate.py '+tmp_savedir2+' '+hkuis_gtdir)
	
	
	if it%snapshot_every==0:
		with open(logfile,'a') as f:
			f.write('iter: %d trainloss: %.4f testloss: %.4f\n' % (it,trainloss,testloss))
			
	
	it = it+1
	#break
