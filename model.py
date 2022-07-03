import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from layers import *
import random
import contextlib
from mydis import *
from resnet import cifar_shakeshake26

def _repeat(num, x):
	if num == 1:
		return x
	shape = list(x.size())
	x = x.unsqueeze(0)
	x = x.expand(num, *shape).contiguous()
	shape[0] = shape[0]*num
	return x.view(shape)

def repeat_tensors(num, mats):
	if torch.is_tensor(mats):
		return _repeat(num, mats)
	else:
		return (_repeat(num, m) for m in mats)


@contextlib.contextmanager
def _disable_tracking_bn_stats(model):

	def disable_attr(m):
		classname = m.__class__.__name__
		if m.training and classname.startswith('BatchNorm'):
			m.momentum = 0

	def enable_attr(m):
		classname = m.__class__.__name__
		if m.training and classname.startswith('BatchNorm'):
			m.momentum = 0.1
			
	model.apply(disable_attr)
	yield
	model.apply(enable_attr)

def l2_normalize(x):
	shape = x.size()
	x = x.view(shape[0], -1)
	x = F.normalize(x, dim=1)
	return x.view(shape)

def mixup(dis_model, x, targets, mixup_alpha=1.0, mixup_start=0, mixup_layers=1):
	layer_mix = random.randint(mixup_start, mixup_start + mixup_layers - 1)
	h_pre = dis_model(x, end_layer=layer_mix-1)
	h, mixed_targets, lam = mixup_data(h_pre, h_pre, targets, mixup_alpha)
	h = dis_model(h, start_layer=layer_mix)
	return h_pre, h, mixed_targets, lam, layer_mix

def mixup_with_adv(dis_model, x, x2, targets, mixup_alpha=1.0, mixup_start=0, mixup_layers=1, mix_toward_adv=None):
	layer_mix = random.randint(mixup_start, mixup_start + mixup_layers - 1)
	h_pre = dis_model(x, end_layer=layer_mix-1)
	h_pre2 = dis_model(x2, end_layer=layer_mix-1)
	h, mixed_targets, lam = mixup_data(h_pre, h_pre2, targets, mixup_alpha, mix_toward_adv)
	h = dis_model(h, start_layer=layer_mix)
	return h_pre, h_pre2, h, mixed_targets, lam, layer_mix

def mixup_data(x, x2, targets, alpha, big_toward_right=None):

	'''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
	lam = np.random.beta(alpha, alpha)
	if big_toward_right is not None:
		if big_toward_right:
			lam = min(lam, 1 - lam)
		else:
			lam = max(lam, 1 - lam)
	
	batch_size = x.size(0)
	index = torch.randperm(batch_size, device=x.device)
	mixed_x = lam * x + (1 - lam) * x2[index]
	mixed_targets = []
	for y in targets:
		if y.dim() > 1:
			mixed_targets.append(lam * y + (1 - lam) * y[index])
		else:
			mixed_targets.append(y[index])
	# y_a, y_b = y, y[index]
	return mixed_x, mixed_targets, lam



def make_discriminator(config):
	if config.d_model=='convl':
		dm = D_CONVL(config.num_classes, config.d_input_noise, config.d_input_noise_method, 
			config.d_drop_hidden, config.d_wn_out, config.d_wn_hid)
	elif config.d_model == 'wrn':
		dm = WRN28_2(config.num_classes, config.d_drop_hidden)
	elif config.d_model=='mlp':
		dm = D_MLP(config.input_size, config.num_classes, config.d_layer_sizes, config.d_input_noise, config.d_hidden_noise, 
			config.d_input_noise_method, config.d_hidden_noise_method, 
			config.d_bn_out, config.d_bn_hid, config.d_wn_out, config.d_wn_hid)
	elif config.d_model == 'res':
		dm = cifar_shakeshake26(config.num_classes)
	else:
		raise ValueError('unsupported discriminator')
	return dm


class GD_model(nn.Module):
	"""docstring for GD_model"""
	def __init__(self, config):
		super(GD_model, self).__init__()
		self.num_classes=config.num_classes
		# self.d_disable_update_bnstat_forp=config.d_disable_update_bnstat_forp
		
		self.dataset = config.dataset
		self.cifar100_zca = config.cifar100_zca
		self.use_ema = config.use_ema
		self.adv_mode = config.adv_mode
		self.adv_num = config.adv_num

		self.l2_dist = config.l2_dist
		self.average_l2 = config.average_l2
		self.vat_xi = config.vat_xi
		self.vat_eps = config.vat_eps
		self.vat_eps_as_bound = config.vat_eps_as_bound
		self.vat_ip = config.vat_ip
		
		self.mixup_mode = config.mixup_mode
		self.mixup_alpha = config.mixup_alpha
		self.mixup_start = config.mixup_start
		self.mixup_layers = config.mixup_layers
		self.mix_toward_adv = config.mix_toward_adv
		
		
		self.dis=make_discriminator(config)
		if self.use_ema:
			self.dis_ema=make_discriminator(config)
		self.init_preprocess_params(config.zca_mean, config.zca_components)
	def init_preprocess_params(self, zca_mean, zca_components):
		# mean and std
		
		if self.dataset == 'cifar10' or (self.dataset == 'cifar100' and self.cifar100_zca):
			self.register_buffer('zca_mean', torch.as_tensor(zca_mean, dtype=torch.float))
			self.register_buffer('zca_components', torch.as_tensor(zca_components, dtype=torch.float))

	def preprocess(self, x):
		
		if self.dataset == 'cifar10' or (self.dataset == 'cifar100' and self.cifar100_zca):
			shape = x.size()
			x = torch.mm(x.view(shape[0], -1) - self.zca_mean, self.zca_components).view(shape)
		return x

	def vat(self, x, target):
		target = target.detach()
		d = torch.randn_like(x)
		with torch.enable_grad():
			with _disable_tracking_bn_stats(self.dis):
				for _ in range(self.vat_ip):
					d = self.vat_xi * l2_normalize(d)
					d.requires_grad_()
					logit = self.dis(x + d)
					if self.l2_dist:
						p = F.softmax(logit, dim=1)
						loss = F.mse_loss(p, target, reduction = 'mean')
						if not self.average_l2:
							loss = loss * self.num_classes
					else:
						logp = F.log_softmax(logit, dim=1)
						loss = F.kl_div(logp, target, reduction='batchmean')
					d = torch.autograd.grad(loss, d)[0]

		cur_eps = (random.uniform(0, 1) * self.vat_eps) if self.vat_eps_as_bound else self.vat_eps
		p = cur_eps * l2_normalize(d)
		return x + p

	def vat_rand(self, x, target):
		target = target.detach()
		d = torch.randn_like(x)
		with torch.enable_grad():
			with _disable_tracking_bn_stats(self.dis):
				d = self.vat_xi * l2_normalize(d)
				x_rand = x + d
				x_rand.requires_grad_()
				logit = self.dis(x_rand)
				if self.l2_dist:
					p = F.softmax(logit, dim=1)
					loss = F.mse_loss(p, target, reduction = 'mean')
					if not self.average_l2:
						loss = loss * self.num_classes
				else:
					logp = F.log_softmax(logit, dim=1)
					loss = F.kl_div(logp, target, reduction='batchmean')
				d2 = torch.autograd.grad(loss, x_rand)[0]

		cur_eps = (random.uniform(0, 1) * self.vat_eps) if self.vat_eps_as_bound else self.vat_eps
		p = cur_eps * l2_normalize(d2)
		return x_rand.detach() + p

	
	
	def train_d(self, x_d, u_d, t, ul_t, temp, combine, sym, with_mixup, with_adv, with_ent, sup=False):
		result = {}
		mixup_mode = self.mixup_mode
		x_d = self.preprocess(x_d)
		x_pred = self.dis(x_d)
		result['x_pred'] = x_pred
		result['t'] = t
		if sup and with_ent:
			result['ent'] = x_pred
		
		if (with_adv or with_mixup or with_ent):
			u_d = x_d if sup else self.preprocess(u_d)
			if combine:
				u_d = torch.cat([x_d, u_d], 0)
				ul_t = torch.cat([t, ul_t], 0)
			result['ul_t'] = ul_t

		if with_adv or with_mixup:
			if sup:
				if sym:
					pred_target = x_pred
				elif not self.use_ema:
					pred_target = x_pred.detach()
				else:
					with torch.no_grad():
						pred_target = self.dis_ema(x_d)
			else:
				if sym:
					pred_target = self.dis(u_d)
					if with_ent:
						result['ent'] = pred_target
				else:
					with torch.no_grad():
						dis_model = self.dis_ema if self.use_ema else self.dis
						pred_target = dis_model(u_d)
			if temp != 1:
				pred_target = pred_target / temp
			pred_target = F.softmax(pred_target, dim=1)
			result['pred_target'] = pred_target

		if with_mixup:
			if mixup_mode == 'rand':
				mixup_mode = random.choice(['rr','ra','aa'])
			elif mixup_mode == 'rand2':
				mixup_mode = random.choice(['rr','ra'])
		
		if with_adv or (with_mixup and (mixup_mode == 'ra' or mixup_mode == 'aa')):
			u_d_repeat, pred_target_repeat, ul_t_repeat = repeat_tensors(self.adv_num, (u_d, pred_target, ul_t))
			if self.adv_mode == 'vat':
				u_adv = self.vat(u_d_repeat, pred_target_repeat)
			elif self.adv_mode == 'vat_rand':
				u_adv = self.vat_rand(u_d_repeat, pred_target_repeat)
			else:
				raise ValueError('unsupported adv mode!')
			if with_adv:
				result['pred_target_adv'] = pred_target_repeat
				result['ul_t_adv'] = ul_t_repeat
			if with_mixup and (mixup_mode == 'ra' or mixup_mode == 'aa'):
				result['ul_t_mix'] = ul_t_repeat
			else:
				result['ul_t_mix'] = ul_t
		elif with_mixup:
			result['ul_t_mix'] = ul_t
		
		if with_mixup:
			
			if mixup_mode == 'rr':
				h_pre, u_mix_pred, (mix_pred_target, mix_t2), lam, layer_mix = mixup(self.dis, u_d, (pred_target, ul_t), self.mixup_alpha, self.mixup_start, self.mixup_layers)
			elif mixup_mode == 'ra':				
				h_pre, h_pre_adv, u_mix_pred, (mix_pred_target, mix_t2), lam, layer_mix = mixup_with_adv(self.dis, u_d_repeat, u_adv, (pred_target_repeat, ul_t_repeat), self.mixup_alpha, self.mixup_start, self.mixup_layers, self.mix_toward_adv)
			elif mixup_mode == 'aa':
				h_pre_adv, u_mix_pred, (mix_pred_target, mix_t2), lam, layer_mix = mixup(self.dis, u_adv, (pred_target_repeat, ul_t_repeat), self.mixup_alpha, self.mixup_start, self.mixup_layers)
			else:
				raise ValueError('unsupported mixup_mode!')
			result['mix'] = (u_mix_pred, mix_pred_target, mix_t2, lam)

		if with_adv:
			with _disable_tracking_bn_stats(self.dis):
				if with_mixup and (mixup_mode == 'ra' or mixup_mode == 'aa'):
					u_adv_pred = self.dis(h_pre_adv, start_layer=layer_mix)
				else:
					u_adv_pred = self.dis(u_adv)
				result['adv'] = u_adv_pred
		
		if with_ent and not ('ent' in result):
			if with_mixup and (mixup_mode == 'rr' or mixup_mode == 'ra'):
				u_pred = self.dis(h_pre, start_layer=layer_mix)
			else:
				u_pred = self.dis(u_d)
			result['ent'] = u_pred

		return result


	


