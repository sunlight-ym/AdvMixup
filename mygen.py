import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from layers import *
import math

class G_CONVL(nn.Module):
	"""docstring for G_CONVL"""
	def __init__(self, input_size, enc_layer, noise_max_norm):
		super(G_CONVL, self).__init__()
		self.layers=nn.ModuleList()
		self.noise_max_norm=noise_max_norm
		self.enc_layer=enc_layer

		input_channels, height, width = input_size
		self.layers.append(conv(input_channels, 64, 4, 2, 1, bn=True, lrelu_a=0))

		self.layers.append(conv(64, 128, 4, 2, 1, bn=True, lrelu_a=0))

		self.layers.append(conv(128, 128, 3, 1, 1, bn=True, lrelu_a=0))

		self.layers.append(conv(128, 128, 3, 1, 1, bn=True, lrelu_a=0))
		
		self.layers.append(deconv(128, 64, 4, 2, 1, bn=True, lrelu_a=0))
		
		self.layers.append(deconv(64, input_channels, 4, 2, 1, bn=False, lrelu_a=nn.Tanh()))

	def encode(self, x):
		h=x
		for layer in self.layers[:self.enc_layer]:
			h=layer(h)
		return h
	def decode(self, h):
		if self.noise_max_norm > 0:
			h = add_noise(h, self.noise_max_norm)
		for layer in self.layers[self.enc_layer:]:
			h=layer(h)
		return h

	def forward(self, x):
		#process x
		# z_old=h
		# h=h.detach()
		# h.requires_grad_()
		# z_new=h
		return self.decode(self.encode(x))

class G_MLP(nn.Module):
	"""docstring for G_MLP"""
	def __init__(self, input_size, enc_layer, noise_max_norm, layer_sizes, out_act, nonlinearity):
		super(G_MLP, self).__init__()
		self.enc_layer=enc_layer
		self.noise_max_norm=noise_max_norm
		self.nonlinearity=getattr(nn, nonlinearity)
		self.input_size=input_size
		flat_input_size = int(np.prod(input_size))

		self.layers=nn.ModuleList()
		layer_sizes=[flat_input_size]+layer_sizes+[flat_input_size]
		num_layers=len(layer_sizes)-1
		
		for i, (m, n) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):			
			act=self.nonlinearity() if i!=num_layers-1 else (nn.Tanh() if out_act else None)
			self.layers.append(linear(m, n, bn=(i!=num_layers-1), act=act))

	def encode(self, x):
		h=x.view(x.size(0), -1)
		for layer in self.layers[:self.enc_layer]:
			h=layer(h)
		return h
	def decode(self, h):
		if self.noise_max_norm > 0:
			h = add_noise(h, self.noise_max_norm)
		for layer in self.layers[self.enc_layer:]:
			h=layer(h)
		return h.view(h.size(0), *self.input_size)

	def forward(self, x):
		#process x
		# z_old=h
		# h=h.detach()
		# h.requires_grad_()
		# z_new=h
		return self.decode(self.encode(x))
		
		

class ResnetBlock(nn.Module):
	def __init__(self, dim, padding_type, use_dropout, dropout_rate):
		super(ResnetBlock, self).__init__()
		self.conv_block = self.build_conv_block(dim, padding_type, use_dropout, dropout_rate)

	def build_conv_block(self, dim, padding_type, use_dropout, dropout_rate):
		conv_block = []
		p = 0
		if padding_type == 'reflect':
			conv_block += [nn.ReflectionPad2d(1)]
		elif padding_type == 'replicate':
			conv_block += [nn.ReplicationPad2d(1)]
		elif padding_type == 'zero':
			p = 1
		else:
			raise NotImplementedError('padding [%s] is not implemented' % padding_type)

		conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=False),
					   nn.BatchNorm2d(dim),
					   nn.ReLU(inplace=True)]
		if use_dropout:
			conv_block += [nn.Dropout(dropout_rate)]

		p = 0
		if padding_type == 'reflect':
			conv_block += [nn.ReflectionPad2d(1)]
		elif padding_type == 'replicate':
			conv_block += [nn.ReplicationPad2d(1)]
		elif padding_type == 'zero':
			p = 1
		else:
			raise NotImplementedError('padding [%s] is not implemented' % padding_type)
		conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=False),
					   nn.BatchNorm2d(dim)]

		return nn.Sequential(*conv_block)

	def forward(self, x):
		out = x + self.conv_block(x)
		return out

class ResnetGenerator(nn.Module):
	def __init__(self, input_size, enc_layer, noise_max_norm, n_downsampling, ngf, kw0, kw1, use_dropout, res_dropout, n_blocks):
		super(ResnetGenerator, self).__init__()
		self.layers=nn.ModuleList()
		self.noise_max_norm=noise_max_norm
		self.enc_layer=enc_layer

		input_nc, height, width = input_size
		self.layers.append(nn.Sequential(nn.ReflectionPad2d(int((kw0-1)/2)),
						 nn.Conv2d(input_nc, ngf, kernel_size=kw0, padding=0, bias=False),
						 nn.BatchNorm2d(ngf),
						 nn.ReLU(inplace=True)))

		# n_downsampling = 2
		for i in range(n_downsampling):
			# mult = 2**i
			in_maps=ngf*(2**i)
			out_maps=in_maps*2
			self.layers.append(nn.Sequential(nn.Conv2d(in_maps, out_maps, kernel_size=kw1, stride=2, padding=math.ceil(kw1/2-1), bias=False),
								  nn.BatchNorm2d(out_maps),
								  nn.ReLU(inplace=True)))

		mult = 2**n_downsampling
		for i in range(n_blocks):
			self.layers.append(ResnetBlock(ngf * mult, padding_type='reflect', use_dropout=use_dropout, dropout_rate=res_dropout))
			
		for i in range(n_downsampling):
			# mult = 2**(n_downsampling - i)
			in_maps=ngf*(2**(n_downsampling-i))
			out_maps=int(in_maps/2)
			self.layers.append(nn.Sequential(nn.ConvTranspose2d(in_maps, out_maps, kernel_size=kw1, stride=2, 
														padding=math.ceil(kw1/2-1), output_padding=(1 if kw1%2!=0 else 0), bias=False),
													nn.BatchNorm2d(out_maps),
													nn.ReLU(inplace=True)))
			
		self.layers.append(nn.Sequential(nn.ReflectionPad2d(int((kw0-1)/2)),
												nn.Conv2d(ngf, input_nc, kernel_size=kw0, padding=0),
												nn.Tanh()))

	def encode(self, x):
		h=x
		for layer in self.layers[:self.enc_layer]:
			h=layer(h)
		return h
	def decode(self, h):
		if self.noise_max_norm > 0:
			h = add_noise(h, self.noise_max_norm)
		for layer in self.layers[self.enc_layer:]:
			h=layer(h)
		return h

	def forward(self, x):
		#process x
		# z_old=h
		# h=h.detach()
		# h.requires_grad_()
		# z_new=h
		return self.decode(self.encode(x))


		