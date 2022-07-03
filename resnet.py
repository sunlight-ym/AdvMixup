import itertools
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function



def cifar_shakeshake26(num_classes):
	model = ResNet32x32(ShakeShakeBlock,
						layers=[4, 4, 4],
						channels=96, num_classes=num_classes,
						downsample='shift_conv')
	# 26 2x96d
	# groups = 1
	# Shake-Even-Image
	return model


class ResNet32x32(nn.Module):
	def __init__(self, block, layers, channels, groups=1, num_classes=1000, downsample='basic'):
		super().__init__()
		assert len(layers) == 3
		self.downsample_mode = downsample
		self.inplanes = 16
		self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1,
							   padding=1, bias=False)
		self.layer1 = self._make_layer(block, channels, groups, layers[0])
		self.layer2 = self._make_layer(
			block, channels * 2, groups, layers[1], stride=2)
		self.layer3 = self._make_layer(
			block, channels * 4, groups, layers[2], stride=2)
		self.avgpool = nn.AvgPool2d(8)
		self.fc1 = nn.Linear(block.out_channels(
			channels * 4, groups), num_classes)
		# self.fc2 = nn.Linear(block.out_channels(
		# 	channels * 4, groups), num_classes)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()

	def _make_layer(self, block, planes, groups, blocks, stride=1):
		downsample = None
		if stride != 1 or self.inplanes != block.out_channels(planes, groups):
			if self.downsample_mode == 'basic' or stride == 1:
				downsample = nn.Sequential(
					nn.Conv2d(self.inplanes, block.out_channels(planes, groups),
							  kernel_size=1, stride=stride, bias=False),
					nn.BatchNorm2d(block.out_channels(planes, groups)),
				)
			elif self.downsample_mode == 'shift_conv':
				downsample = ShiftConvDownsample(in_channels=self.inplanes,
												 out_channels=block.out_channels(planes, groups))
			else:
				assert False

		layers = []
		layers.append(block(self.inplanes, planes, groups, stride, downsample))
		self.inplanes = block.out_channels(planes, groups)
		for i in range(1, blocks):
			layers.append(block(self.inplanes, planes, groups))

		return nn.Sequential(*layers)

	def forward(self, x, start_layer=-1, end_layer=None):
		if end_layer == -1:
			return x
		if start_layer <= 0:
			x = self.conv1(x)
			x = self.layer1(x)
			if end_layer == 0:
				return x
		if start_layer <= 1:
			x = self.layer2(x)
			if end_layer == 1:
				return x
		if start_layer <= 2:
			x = self.layer3(x)
			if end_layer == 2:
				return x
		if start_layer <= 3:
			x = self.avgpool(x)
			x = x.view(x.size(0), -1)
			x = self.fc1(x)#, self.fc2(x)
		return x


def conv3x3(in_planes, out_planes, stride=1):
	"3x3 convolution with padding"
	return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
					 padding=1, bias=False)




class ShakeShakeBlock(nn.Module):
	@classmethod
	def out_channels(cls, planes, groups):
		assert groups == 1
		return planes

	def __init__(self, inplanes, planes, groups, stride=1, downsample=None):
		super().__init__()
		assert groups == 1
		self.conv_a1 = conv3x3(inplanes, planes, stride)
		self.bn_a1 = nn.BatchNorm2d(planes)
		self.conv_a2 = conv3x3(planes, planes)
		self.bn_a2 = nn.BatchNorm2d(planes)

		self.conv_b1 = conv3x3(inplanes, planes, stride)
		self.bn_b1 = nn.BatchNorm2d(planes)
		self.conv_b2 = conv3x3(planes, planes)
		self.bn_b2 = nn.BatchNorm2d(planes)

		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		a, b, residual = x, x, x

		a = F.relu(a, inplace=False)
		a = self.conv_a1(a)
		a = self.bn_a1(a)
		a = F.relu(a, inplace=True)
		a = self.conv_a2(a)
		a = self.bn_a2(a)

		b = F.relu(b, inplace=False)
		b = self.conv_b1(b)
		b = self.bn_b1(b)
		b = F.relu(b, inplace=True)
		b = self.conv_b2(b)
		b = self.bn_b2(b)

		ab = shake(a, b, training=self.training)

		if self.downsample is not None:
			residual = self.downsample(x)

		return residual + ab


class Shake(Function):
	@classmethod
	def forward(cls, ctx, inp1, inp2, training):
		assert inp1.size() == inp2.size()
		gate_size = [inp1.size()[0], *itertools.repeat(1, inp1.dim() - 1)]
		gate = inp1.new(*gate_size)
		if training:
			gate.uniform_(0, 1)
		else:
			gate.fill_(0.5)
		return inp1 * gate + inp2 * (1. - gate)

	@classmethod
	def backward(cls, ctx, grad_output):
		grad_inp1 = grad_inp2 = grad_training = None
		gate_size = [grad_output.size()[0], *itertools.repeat(1,
															  grad_output.dim() - 1)]
		gate = grad_output.data.new(*gate_size).uniform_(0, 1)
		if ctx.needs_input_grad[0]:
			grad_inp1 = grad_output * gate
		if ctx.needs_input_grad[1]:
			grad_inp2 = grad_output * (1 - gate)
		assert not ctx.needs_input_grad[2]
		return grad_inp1, grad_inp2, grad_training


def shake(inp1, inp2, training=False):
	return Shake.apply(inp1, inp2, training)


class ShiftConvDownsample(nn.Module):
	def __init__(self, in_channels, out_channels):
		super().__init__()
		self.relu = nn.ReLU(inplace=True)
		self.conv = nn.Conv2d(in_channels=2 * in_channels,
							  out_channels=out_channels,
							  kernel_size=1,
							  groups=2)
		self.bn = nn.BatchNorm2d(out_channels)

	def forward(self, x):
		x = torch.cat((x[:, :, 0::2, 0::2],
					   x[:, :, 1::2, 1::2]), dim=1)
		x = self.relu(x)
		x = self.conv(x)
		x = self.bn(x)
		return x

