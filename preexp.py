import argparse
import time
import os
import sys
import numpy as np
import random
import torch
import torch.nn.functional as F
from data_loader import load_data
from utils import *
from adv import AEG
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def str2bool(v):
	return v.lower() in ('true')

parser=argparse.ArgumentParser()
parser.add_argument('v', type=str)
parser.add_argument('-seed', type=int, default=1)
parser.add_argument('-load_ready', type=str2bool, default=False)
parser.add_argument('-model_path', type=str, default=None)
parser.add_argument('-source_model_path', type=str, default=None, nargs='?')
parser.add_argument('-gpu', type=int, default=0)
parser.add_argument('-dataset', type=str, default='cifar')
parser.add_argument('-data_path', type=str, default='../datasets')
parser.add_argument('-save_dir', type=str, default='/sata/yunma3/MIX')
parser.add_argument('-batch_size', type=int, default=100)
parser.add_argument('-grad_on_prob', type=str2bool, default=False)
parser.add_argument('-grad_on_vec', type=str2bool, default=False)

parser.add_argument('-adv_mode', type=str, default='vat')
parser.add_argument('-epsilon', type=float, default=2)
parser.add_argument('-max_norm', type=str2bool, default=False, nargs='?')
parser.add_argument('-step_epsilon', type=float, default=2, nargs='?')
parser.add_argument('-xi', type=float, default=1e-6, nargs='?')
parser.add_argument('-n_iter', type=int, default=1, nargs='?')
parser.add_argument('-l2_dist', type=str2bool, default=False, nargs='?')

config=parser.parse_args()
print(' '.join(sys.argv))
device = torch.device('cuda', config.gpu)
save_path=os.path.join(config.save_dir, config.dataset, 'analy', config.v)
makedirs(save_path)

random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed(config.seed)

def mixup_data(x, x2, targets, lam):

	'''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
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
	return mixed_x, mixed_targets
def mixup_acc(y, t1, t2):
	with torch.no_grad():
		pred = torch.argmax(y, dim=1)
		acc = (pred==t1) | (pred==t2)
		acc = acc.float().sum().item()
	return acc

def eval_mixpath(data_loader, model, source_model, adv_generator):
	start_time=time.time()
	n_total=len(data_loader.dataset)
	dis_model = model.dis_ema if model.use_ema else model.dis
	lam_arr = np.arange(0, 1.01, 0.05)
	acc = np.zeros(lam_arr.shape)
	grad_norm = np.zeros(lam_arr.shape)
	for x, t in data_loader:
		x, t = to_device(device, (x, t))
		x = model.preprocess(x)
		if adv_generator is not None:
			x2 = adv_generator(x, t, source_model.dis_ema if source_model.use_ema else source_model.dis)
		else:
			x2 = x
		cur_acc = np.zeros(lam_arr.shape)
		cur_grad_norm = np.zeros(lam_arr.shape)
		for i, lam in enumerate(lam_arr):
			mixed_x, (t2,) = mixup_data(x, x2, (t,), lam)
			mixed_x.requires_grad_()
			mixed_y = dis_model(mixed_x)
			cur_acc[i] = mixup_acc(mixed_y, t, t2)
			if config.grad_on_prob:
				mixed_y = F.softmax(mixed_y, dim=1)
			if config.grad_on_vec:
				grad = torch.autograd.grad(mixed_y, mixed_x, torch.ones_like(mixed_y))[0]
			else:
				mixed_y = mixed_y.max(1)[0]
				grad = torch.autograd.grad(mixed_y, mixed_x, torch.ones_like(mixed_y))[0]
			cur_grad_norm[i] = torch.norm(grad.view(grad.size(0), -1), p=2, dim=1).sum().item()


		acc = acc + cur_acc
		grad_norm = grad_norm + cur_grad_norm
	acc = acc / n_total
	grad_norm = grad_norm / n_total
	print('Finish evaluating {} in {:6.0f} seconds'.format('pure mix' if adv_generator is None else 'adv mix', time.time()-start_time))
	return acc, grad_norm

def draw(data1, data2, name1, name2, fname, y_label, y_lim=None, y_ticks=None):
	lam_arr = np.arange(0, 1.01, 0.05)
	fig, ax=plt.subplots()
	ax.plot(lam_arr, data1, label=name1)
	ax.plot(lam_arr, data2, label=name2)
	ax.set_xlabel(r'$\lambda$')
	ax.set_ylabel(y_label)
	if y_lim is not None:
		ax.set_ylim(*y_lim)
	if y_ticks is not None:
		ax.set_yticks(y_ticks)
	ax.legend(loc='upper right')

	fig.savefig(fname)
	plt.close(fig)


if __name__=='__main__':
	if not config.load_ready:
		data_path = os.path.join(config.data_path, config.dataset, 'data')
		test_loader = load_data(config.dataset, data_path, 'd', False, 
			config.batch_size, config.batch_size, config.batch_size, True, 0, 
			-1, 0, examples_per_class=0)['test']

		model = torch.load(config.model_path, map_location=device)['model']
		frozen_model(model)
		model.eval()

		if config.source_model_path is not None:
			source_model = torch.load(config.source_model_path, map_location=device)['model']
			frozen_model(source_model)
			source_model.eval()
		else:
			source_model = model

		acc1, gn1 = eval_mixpath(test_loader, model, source_model, None)
		adv_generator = AEG(config.adv_mode, config.epsilon, config.max_norm, config.step_epsilon, config.xi, config.n_iter, config.l2_dist)
		acc2, gn2 = eval_mixpath(test_loader, model, source_model, adv_generator)
		np.save(os.path.join(save_path, 'acc1.npy'), acc1)
		np.save(os.path.join(save_path, 'acc2.npy'), acc2)
		np.save(os.path.join(save_path, 'gn1.npy'), gn1)
		np.save(os.path.join(save_path, 'gn2.npy'), gn2)
	else:
		acc1 = np.load(os.path.join(save_path, 'acc1.npy'))
		acc2 = np.load(os.path.join(save_path, 'acc2.npy'))
		gn1 = np.load(os.path.join(save_path, 'gn1.npy'))
		gn2 = np.load(os.path.join(save_path, 'gn2.npy'))

	acc_file = os.path.join(save_path, 'acc.pdf')
	gn_file = os.path.join(save_path, 'gn.pdf')
	draw((1-acc1)*100, (1-acc2)*100, 'mix', 'advmix', acc_file, 'error rate (%)', (10, 40))
	draw(gn1, gn2, 'mix', 'advmix', gn_file, 'gradient norm')



