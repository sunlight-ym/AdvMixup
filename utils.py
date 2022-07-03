#from __future__ import division
import torch
# from torch.autograd import Variable
import numpy as np
import math
import os
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# from matplotlib.colors import ListedColormap
# from scipy.misc import imsave
def makedirs(path):
	if not os.path.exists(path):
		os.makedirs(path)


def rampup(cur_step, rampup_iters_end, rampup_iters_start=0, rampup_alpha=5.0, decay_step=1):
	if decay_step>1:
		cur_step=math.ceil(cur_step/decay_step)
		rampup_iters_start=rampup_iters_start//decay_step
		rampup_iters_end=rampup_iters_end//decay_step
	if cur_step<=rampup_iters_start:
		return 0.0
	elif cur_step<rampup_iters_end:
		p=1-max(0.0, (cur_step-rampup_iters_start)/(rampup_iters_end-rampup_iters_start))
		return math.exp(-p*p*rampup_alpha)
	else:
		return 1.0
def rampdown(cur_step, rampdown_iters_end, rampdown_iters_start, rampdown_alpha=12.5, decay_step=1):
	if decay_step>1:
		cur_step=math.ceil(cur_step/decay_step)
		rampdown_iters_start=rampdown_iters_start//decay_step
		rampdown_iters_end=rampdown_iters_end//decay_step
	if cur_step<=rampdown_iters_start:
		return 1.0
	elif cur_step<rampdown_iters_end:
		p=(cur_step-rampdown_iters_start)/(rampdown_iters_end-rampdown_iters_start)
		return math.exp(-p*p*rampdown_alpha)
	else:
		return 0.0

def cosine_rampdown(current, rampdown_length):
	"""Cosine rampdown from https://arxiv.org/abs/1608.03983"""
	assert 0 <= current <= rampdown_length
	return float(.5 * (np.cos(np.pi * current / rampdown_length) + 1))

def init_accl(keys):
	a={}
	for k in keys:
		a[k]=0
	return a

def inc_accl(a, result):
	for k in a:
		a[k] += result[k]
def div_accl(a, c):
	for k in a:
		a[k] /= c

def to_device(device, mats):
	if torch.is_tensor(mats):
		return mats.to(device)
	else:
		return (to_device(device, m) for m in mats)

def frozen_model(model):
	for param in model.parameters():
		param.requires_grad = False