import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from layers import *
from torch.nn.utils import weight_norm


class D_CONVL(nn.Module):
	"""docstring for D_CONVL"""
	def __init__(self, num_classes, input_noise, input_noise_method, drop_hidden, wn_out, wn_hid):
		super(D_CONVL, self).__init__()
		# self.input_size = input_size
		self.num_classes = num_classes
		self.apply_noise=(input_noise>0)

		if self.apply_noise:
			if input_noise_method=='drop':
				self.preprocess=nn.Dropout(input_noise)
			elif input_noise_method=='gaussian':
				self.preprocess=GaussianNoise(input_noise)
			else:
				raise ValueError('unsupported noise method')

		self.conv_block1=nn.Sequential(OrderedDict([
			('conv11', conv(3, 128, 3, 1, 1, bn=True, wn=wn_hid, lrelu_a=0.1)),
			('conv12', conv(128, 128, 3, 1, 1, bn=True, wn=wn_hid, lrelu_a=0.1)),
			('conv13', conv(128, 128, 3, 1, 1, bn=True, wn=wn_hid))
		]))
		self.downsamp_1=nn.Sequential(OrderedDict([
			('pool1', nn.MaxPool2d(2, 2)),
			('drop1', nn.Dropout(drop_hidden))
		]))
		self.conv_block2=nn.Sequential(OrderedDict([
			('conv21', conv(128, 256, 3, 1, 1, bn=True, wn=wn_hid, lrelu_a=0.1)),
			('conv22', conv(256, 256, 3, 1, 1, bn=True, wn=wn_hid, lrelu_a=0.1)),
			('conv23', conv(256, 256, 3, 1, 1, bn=True, wn=wn_hid))
		]))
		self.downsamp_2=nn.Sequential(OrderedDict([
			('pool1', nn.MaxPool2d(2, 2)),
			('drop1', nn.Dropout(drop_hidden))
		]))
		self.conv_block3=nn.Sequential(OrderedDict([
			('conv31', conv(256, 512, 3, 1, 0, bn=True, wn=wn_hid, lrelu_a=0.1)),
			('conv32', conv(512, 256, 1, 1, 0, bn=True, wn=wn_hid, lrelu_a=0.1)),
			('conv33', conv(256, 128, 1, 1, 0, bn=True, wn=wn_hid))
		]))
		
		self.activation = nn.LeakyReLU(0.1)

		last_layer=[]
		last_linear = nn.Linear(128, self.num_classes)
		if wn_out:
			last_layer.append(('linear', weight_norm(last_linear)))
		else:
			last_layer.append(('linear', last_linear))
			last_layer.append(('bn', nn.BatchNorm1d(self.num_classes)))
		
		self.last_layer=nn.Sequential(OrderedDict(last_layer))

	

	def forward(self, h, start_layer=-1, end_layer=None):
		if end_layer == -1:
			return h
		if start_layer <= 0:
			if self.apply_noise:
				h=self.preprocess(h)
			h=self.conv_block1(h)
			if end_layer == 0:
				return h
		if start_layer <= 1:
			h=self.activation(h)
			h=self.downsamp_1(h)
			h=self.conv_block2(h)
			if end_layer == 1:
				return h
		if start_layer <= 2:
			h=self.activation(h)
			h=self.downsamp_2(h)
			h=self.conv_block3(h)
			if end_layer == 2:
				return h
		if start_layer <= 3:
			h=self.activation(h)
			h=h.view(h.size(0), h.size(1), -1).mean(2)
			h=self.last_layer(h)
		return h

class D_MLP(nn.Module):
	"""docstring for D_MLP"""
	def __init__(self, input_size, num_classes, layer_sizes, input_noise, hidden_noise, input_noise_method, hidden_noise_method, 
		bn_out, bn_hid, wn_out, wn_hid):
		super(D_MLP, self).__init__()
		# self.input_size = input_size
		self.num_classes = num_classes
		self.activation=nn.ReLU()
		
		self.input_size=int(np.prod(input_size))
		layer_sizes = [self.input_size]+layer_sizes+[num_classes]
		num_layers=len(layer_sizes)-1
		self.layers=nn.ModuleList()
		
		for i, (m, n) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
			in_noise=input_noise if i==0 else hidden_noise
			noise_method=input_noise_method if i==0 else hidden_noise_method
			if in_noise>0:
				if noise_method=='gaussian':
					noise_method=GaussianNoise
				elif noise_method=='drop':
					noise_method=nn.Dropout
				else:
					raise ValueError('unsupported noise method')
			bn = bn_hid if i!=num_layers-1 else bn_out
			wn = wn_hid if i!=num_layers-1 else wn_out
			self.layers.append(linear(m, n, in_noise, noise_method, bn=bn, wn=wn))
			
	
	def forward(self, h, start_layer=-1, end_layer=None):
		if start_layer == -1:
			h=h.view(h.size(0), -1)
			start_layer = 0
		if end_layer == -1:
			return h

		for i, layer in enumerate(self.layers[start_layer:]):
			if i + start_layer != 0:
				h = self.activation(h)
			h=layer(h)
			if end_layer == i + start_layer:
				return h
		return h


def conv3x3(in_planes, out_planes, stride=1):
	return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)

def conv_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		init.xavier_uniform(m.weight, gain=np.sqrt(2))
		init.constant(m.bias, 0)
	elif classname.find('BatchNorm') != -1:
		init.constant(m.weight, 1)
		init.constant(m.bias, 0)

class wide_basic(nn.Module):
	def __init__(self, in_planes, planes, dropout_rate, stride=1):
		super(wide_basic, self).__init__()
		self.bn1 = nn.BatchNorm2d(in_planes)
		self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
		self.dropout = nn.Dropout(p=dropout_rate)
		self.bn2 = nn.BatchNorm2d(planes)
		self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)
		self.act = nn.LeakyReLU()

		self.shortcut = nn.Sequential()
		if stride != 1 or in_planes != planes:
			self.shortcut = nn.Sequential(
				nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
			)

	def forward(self, x):
		out = self.dropout(self.conv1(self.act(self.bn1(x))))
		out = self.conv2(self.act(self.bn2(out)))
		out += self.shortcut(x)

		return out

class Wide_ResNet(nn.Module):
	
	def __init__(self, depth, widen_factor,dropout_rate, num_classes):
		super(Wide_ResNet, self).__init__()
		self.in_planes = 16
		self.act = nn.LeakyReLU()

		assert ((depth-4)%6 ==0), 'Wide-resnet_v2 depth should be 6n+4'
		n = int((depth-4)/6)
		k = widen_factor

		print('| Wide-Resnet %dx%d' %(depth, k))
		nStages = [16, 16*k, 32*k, 64*k]

		self.conv1 = conv3x3(3,nStages[0])
		self.layer1 = self._wide_layer(wide_basic, nStages[1], n, dropout_rate, stride=1)
		self.layer2 = self._wide_layer(wide_basic, nStages[2], n, dropout_rate, stride=2)
		self.layer3 = self._wide_layer(wide_basic, nStages[3], n, dropout_rate, stride=2)
		self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
		self.linear = nn.Linear(nStages[3], num_classes)

	def _wide_layer(self, block, planes, num_blocks,dropout_rate, stride):
		strides = [stride] + [1]*(num_blocks-1)
		layers = []

		for stride in strides:
			layers.append(block(self.in_planes, planes, dropout_rate, stride))
			self.in_planes = planes

		return nn.Sequential(*layers)
	
		
		
	def forward(self, out, start_layer=-1, end_layer=None):
		if end_layer == -1:
			return out
		if start_layer <= 0:
			out = self.conv1(out)
			out = self.layer1(out)
			if end_layer == 0:
				return out
		if start_layer <= 1:
			out = self.layer2(out)
			if end_layer == 1:
				return out
		if start_layer <= 2:
			out = self.layer3(out)
			if end_layer == 2:
				return out
		
		if start_layer <= 3:
			out = self.act(self.bn1(out))
			out = F.avg_pool2d(out, 8)
			out = out.view(out.size(0), -1)
			out = self.linear(out)
		
		return out

		
		
		
	
def WRN28_10(num_classes=10, dropout = 0.0):
	model = Wide_ResNet(depth=28, widen_factor=10, dropout_rate = dropout, num_classes=num_classes)
	return model

def WRN28_2(num_classes=10, dropout = 0.0):
	model = Wide_ResNet(depth =28, widen_factor =2,dropout_rate = dropout, num_classes = num_classes)
	return model
