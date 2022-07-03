import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import math
from torch.nn.utils import weight_norm


def truncated_normal_(tensor, mean=0, std=1):
	size = tensor.shape
	tmp = tensor.new_empty(size + (4,)).normal_()
	valid = (tmp < 2) & (tmp > -2)
	ind = valid.max(-1, keepdim=True)[1]
	tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
	tensor.data.mul_(std).add_(mean)

def add_noise(x, noise_max_norm):
	noise = x.new_zeros(x.size())
	truncated_normal_(noise, std=noise_max_norm)
	return x + noise


def conv(c_in, c_out, k_size, stride, pad, bn=True, wn=False, lrelu_a=None):
	"""Custom convolutional layer for simplicity."""
	layers = []
	if wn:
		layers.append(('conv', weight_norm(nn.Conv2d(c_in, c_out, k_size, stride, pad))))
	else:
		layers.append(('conv', nn.Conv2d(c_in, c_out, k_size, stride, pad)))
	if bn:
		layers.append(('bn', nn.BatchNorm2d(c_out)))
	if lrelu_a is not None:
		layers.append(('lrelu', nn.LeakyReLU(lrelu_a) if lrelu_a>0 else nn.ReLU()))
	return nn.Sequential(OrderedDict(layers))

def linear(h_in, h_out, in_noise=0, noise_method=None, bn=True, wn=False, act=None):
	"""Custom convolutional layer for simplicity."""
	layers = []
	if in_noise>0:
		layers.append(('noise', noise_method(in_noise)))
	if wn:
		layers.append(('linear', weight_norm(nn.Linear(h_in, h_out))))
	else:
		layers.append(('linear', nn.Linear(h_in, h_out)))
	if bn:
		layers.append(('bn', nn.BatchNorm1d(h_out)))
	if act is not None:
		layers.append(('act', act))
	return nn.Sequential(OrderedDict(layers))

def deconv(c_in, c_out, k_size, stride, pad, bn=True, wn=False, lrelu_a=None):
	"""Custom convolutional layer for simplicity."""
	layers = []
	if wn:
		layers.append(('deconv', weight_norm(nn.ConvTranspose2d(c_in, c_out, k_size, stride, pad))))
	else:
		layers.append(('deconv', nn.ConvTranspose2d(c_in, c_out, k_size, stride, pad)))
	if bn:
		layers.append(('bn', nn.BatchNorm2d(c_out)))
	if lrelu_a is not None:
		if type(lrelu_a)==float:
			if lrelu_a == 0:
				act=nn.ReLU()
			elif lrelu_a > 0:
				act=nn.LeakyReLU(lrelu_a)
		else:
			act=lrelu_a
		layers.append(('lrelu', act))

	return nn.Sequential(OrderedDict(layers))


class GaussianNoise(nn.Module):
	"""docstring for GaussianNoise"""
	def __init__(self, sigma):
		super(GaussianNoise, self).__init__()
		self.sigma = sigma
	def forward(self, x):
		if self.training and self.sigma>0:
			noise = x.new_empty(x.size()).normal_(std=self.sigma)
			return x+noise
		else:
			return x
