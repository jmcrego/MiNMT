# -*- coding: utf-8 -*-

import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
from transformer.Model import AddPositionalEncoding, Encoder_Decoder
from transformer.Optimizer import OptScheduler
from transformer.Model import prepare_source, prepare_target


def plotPoints2d(X,Y,xlabel=None,ylabel=None,legend=None,f=None):
  plt.figure(figsize=(15, 5))
  plt.plot(X,Y)
  if xlabel is not None:
    plt.xlabel(xlabel)
  if ylabel is not None:
    plt.ylabel(ylabel)
  if legend is not None:
    plt.legend(legend)
  if f is not None:
    plt.savefig(f)
  plt.grid(True)
  plt.show()

def plotMatrix2d(X,f=None):
  #print(X.shape)
  #print(X)
  plt.figure(figsize=(10, 10))
  if f is not None:
    plt.savefig(f)
  plt.imshow(X)

def plotPositionalEncoding():
  bs = 1
  ls = 100
  ed = 20
  plt.figure(figsize=(20, 5))
  pe = AddPositionalEncoding(ed, 0.0)               #ed, dropout
  y = pe.forward(Variable(torch.zeros(bs, ls, ed))) #[bs, ls, ed]
  y = y[0, :, 4:8].data.numpy()                     #[ls, 4]
  plt.plot(np.arange(ls), y)
  plt.grid(True)
  plt.legend(["dim %d"%p for p in [4,5,6,7]])
  plt.show()

def plotLRate(N):
  emb_dim = 256
  noam_scale = 2.0
  noam_warmup = 4000
  lr = 2.0
  beta1 = 0.9
  beta2 = 0.998
  eps = 1e-9
  n_layers = 6
  n_heads = 8
  ff_dim = 1024
  qk_dim = 64
  v_dim = 64
  dropout = 0.1
  Vs = 32000
  Vt = 32000
  idx_pad = 0
  model = Encoder_Decoder(n_layers, ff_dim, n_heads, emb_dim, qk_dim, v_dim, dropout, Vs, Vt, idx_pad)
  optim = torch.optim.Adam(model.parameters(), lr=lr, betas=(beta1, beta2), eps=eps)
  optScheduler = OptScheduler(optim, emb_dim, noam_scale, noam_warmup, 0)
  X = [i for i in range(1,N)]
  Y = [optScheduler.lrate(i) for i in range(1,N)]
  xlabel = '#Iter'
  ylabel = 'LRate'
  legend = ["dim={} scale={:.2f} warmup={}".format(emb_dim,noam_scale,noam_warmup)]
  file = None
  plotPoints2d(X, Y, xlabel=xlabel, ylabel=ylabel, legend=legend, f=file)
  

def plotCurves(file):
	step_s = []
	lr_s = []
	loss_s = []
	vstep_s = []
	vloss_s = []
	with open(file,'r') as f: 
		for l in f:
    	#[2021-01-12_12:07:11.086] INFO Learning step:363700 epoch:18 batch:4813/21111 ms/batch:213.41 lr:0.000147 loss/tok:2.486
			step = None
			lr = None
			loss = None
			toks = l.rstrip().split()
			if toks[2] == 'LearningStep:' and toks[12] == 'lr:' and toks[14] == 'loss:':
				step_s.append(int(toks[3]))
				lr_s.append(float(toks[13]))
				loss_s.append(float(toks[15]))
			elif toks[2] == 'Validation' and toks[3] == 'LearningStep:' and toks[9] == 'loss:':
				vstep_s.append(int(toks[4]))
				vloss_s.append(float(toks[10]))

	plt.figure(figsize=(20, 10))

	plt.subplot(211)
	plt.plot(np.asarray(step_s), np.asarray(loss_s))
	plt.legend(["Learning Curve"])
	plt.xlabel("#Steps")
	plt.ylabel("Loss")
	plt.grid(True)

	plt.subplot(212)
	plt.plot(np.asarray(vstep_s), np.asarray(vloss_s))
	plt.legend(["Validation Curve"])
	plt.xlabel("#Steps")
	plt.ylabel("Loss")
	plt.grid(True)

#	plt.subplot(213)
#	plt.plot(np.asarray(step_s), np.asarray(lr_s))
#	plt.legend(["Learning Rate"])
#	plt.xlabel("#Steps")
#	plt.ylabel("Rate")
#	plt.grid(True)

	plt.show()

def plotMasks():
	batch_src = [[1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7, 8, 9]]
	batch_tgt = [[1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 5, 6]]
	src, tgt, ref, msk_src, msk_tgt = prepare_source(batch_src, batch_tgt, 0, 'cpu')
	print('src',src)
	print('tgt',tgt)
	print('ref',ref)
	print('msk_src = {}'.format(msk_src.shape),msk_src)
	print('msk_tgt = {}'.format(msk_tgt.shape),msk_tgt)

	plt.figure(figsize=(5,5))

	plt.subplot(211)
	plt.imshow(msk_src.squeeze().data.numpy())

	plt.subplot(212)
	plt.imshow(msk_tgt[0].data.numpy())

	plt.show()


if __name__ == '__main__':

  #plotPositionalEncoding()
  #plotLRate(1000000)
  #plotMasks()
  plotCurves(sys.argv[1])
