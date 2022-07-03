#from __future__ import division, print_function
import argparse
import os, time
import random
import torch
import numpy as np
import sys
from solver_gd import Solver_GD


def str2bool(v):
	return v.lower() in ('true')
def str2intlist(v):
	return list(map(int, v.split(',')))
def str2floatlist(v):
	return list(map(float, v.split(',')))
def str2floattuple(v):
	return tuple(map(float, v.split(',')))
	
if __name__=='__main__':

	parser=argparse.ArgumentParser()

	#data parameters
	parser.add_argument('v', type=str)
	parser.add_argument('-save_only', type=str2bool, default=False)
	parser.add_argument('-seed', type=int, default=-1)
	parser.add_argument('-gpu', type=int, default=0)
	parser.add_argument('-workers', type=int, default=0)
	parser.add_argument('-benchmark', type=str2bool, default=True)
	parser.add_argument('-g_only', type=str2bool, default=False)
	parser.add_argument('-cifar100_zca', type=str2bool, default=True, nargs='?')
	parser.add_argument('-dataset', type=str, default='cifar')
	parser.add_argument('-save_dir', type=str, default='/sata/yunma3/MIX')
	parser.add_argument('-data_path', type=str, default='../datasets')
	parser.add_argument('-batch_size', type=int, default=32)
	parser.add_argument('-ul_batch_size', type=int, default=128, nargs='?')
	parser.add_argument('-eval_batch_size', type=int, default=128)
	parser.add_argument('-valid_labels_per_class', type=int, default=0)
	parser.add_argument('-labels_per_class', type=int, default=400)
	parser.add_argument('-disjoint', type=str2bool, default=False)
	parser.add_argument('-data_aug', type=str2bool, default=False)

	parser.add_argument('-g_model', type=str, default='res', nargs='?')
	parser.add_argument('-g_enc_layer', type=int, default=5, nargs='?')
	parser.add_argument('-g_noise_max_norm', type=float, default=0, nargs='?')
	parser.add_argument('-g_out_act', type=str2bool, default=True, nargs='?')
	parser.add_argument('-g_layer_sizes', type=str2intlist, default=[1000, 500, 250, 500, 1000], nargs='?')
	parser.add_argument('-g_nonlinearity', type=str, default='ReLU', nargs='?')
	parser.add_argument('-g_ngf', type=int, default=16, nargs='?')
	parser.add_argument('-g_kw0', type=int, default=3, nargs='?')
	parser.add_argument('-g_kw1', type=int, default=3, nargs='?')
	parser.add_argument('-g_use_dropout', type=str2bool, default=True, nargs='?')
	parser.add_argument('-g_res_dropout', type=float, default=0.5, nargs='?')
	parser.add_argument('-g_n_blocks', type=int, default=2, nargs='?')
	parser.add_argument('-g_n_downsampling', type=int, default=2, nargs='?')

	parser.add_argument('-d_model', type=str, default='convl', nargs='?')
	parser.add_argument('-d_input_noise', type=float, default=0.0, nargs='?')
	parser.add_argument('-d_input_noise_method', type=str, default='gaussian', nargs='?')
	parser.add_argument('-d_wn_out', type=str2bool, default=False, nargs='?')
	parser.add_argument('-d_wn_hid', type=str2bool, default=False, nargs='?')
	parser.add_argument('-d_drop_hidden', type=float, default=0.5, nargs='?')
	parser.add_argument('-d_layer_sizes', type=str2intlist, default=[1000, 500, 250, 250, 250], nargs='?')
	parser.add_argument('-d_hidden_noise', type=float, default=0.0, nargs='?')
	parser.add_argument('-d_hidden_noise_method', type=str, default='drop', nargs='?')
	parser.add_argument('-d_bn_out', type=str2bool, default=False, nargs='?')
	parser.add_argument('-d_bn_hid', type=str2bool, default=False, nargs='?')
	
	parser.add_argument('-nll_weight', type=float, default=1, nargs='?')
	parser.add_argument('-adv_weight', type=float, default=1, nargs='?')
	parser.add_argument('-mixup_weight', type=float, default=1, nargs='?')
	parser.add_argument('-entropy_weight', type=float, default=1, nargs='?')
	parser.add_argument('-reconst_weight', type=float, default=1, nargs='?')
	parser.add_argument('-g_adv_weight', type=float, default=1, nargs='?')
	parser.add_argument('-weight_rampup_iters_start', type=int, default=1, nargs='?')
	parser.add_argument('-weight_rampup_iters_end', type=int, default=1, nargs='?')
	
	parser.add_argument('-adv_mode', type=str, default='vat', nargs='?')
	parser.add_argument('-adv_num', type=int, default=1, nargs='?')
	parser.add_argument('-vat_ip', type=int, default=1, nargs='?')
	parser.add_argument('-vat_xi', type=float, default=1e-6, nargs='?')
	parser.add_argument('-vat_eps', type=float, default=2, nargs='?')
	parser.add_argument('-vat_eps_as_bound', type=str2bool, default=False, nargs='?')
	parser.add_argument('-use_ema', type=str2bool, default=True, nargs='?')
	parser.add_argument('-ema_decay', type=float, default=0.999, nargs='?')
	parser.add_argument('-mixup_mode', type=str, default='rr', nargs='?')
	parser.add_argument('-mixup_ae_update_g', type=str2bool, default=False, nargs='?')
	parser.add_argument('-mixup_alpha', type=float, default=1, nargs='?')
	parser.add_argument('-mixup_start', type=int, default=0, nargs='?')
	parser.add_argument('-mixup_layers', type=int, default=1, nargs='?')
	parser.add_argument('-mix_toward_adv', type=str2bool, default=None, nargs='?')
	parser.add_argument('-g_mixup', type=str2bool, default=False, nargs='?')
	parser.add_argument('-average_l2', type=str2bool, default=False, nargs='?')
	parser.add_argument('-l2_dist', type=str2bool, default=False, nargs='?')
	parser.add_argument('-sym', type=str2bool, default=False, nargs='?')
	parser.add_argument('-sym_start_at', type=int, default=0, nargs='?')
	parser.add_argument('-detach_enc_for_reconst_only', type=str2bool, default=True, nargs='?')
	parser.add_argument('-temperature', type=float, default=1, nargs='?')
	parser.add_argument('-temp_rampdown_iters_end', type=int, default=None, nargs='?')
	parser.add_argument('-temp_rampdown_iters_start', type=int, default=None, nargs='?')
	parser.add_argument('-temp_final', type=float, default=1e-3, nargs='?')
	
	#------optimization
	
	parser.add_argument('-optim_method', type=str, default='adam')
	parser.add_argument('-lr_rampup_iters_end', type=int, default=1)
	parser.add_argument('-lr_rampup_iters_start', type=int, default=0)
	
	parser.add_argument('-lr_g', type=float, default=0.003, nargs='?')
	parser.add_argument('-beta1_g', type=float, default=0.5, nargs='?')
	parser.add_argument('-beta2_g', type=float, default=0.999, nargs='?')
	parser.add_argument('-decay_rate_g', type=float, default=0.9, nargs='?')
	parser.add_argument('-weight_decay_g', type=float, default=0.0, nargs='?')
	parser.add_argument('-lr_d', type=float, default=0.0003, nargs='?')
	parser.add_argument('-beta1_d', type=float, default=0.5, nargs='?')
	parser.add_argument('-beta2_d', type=float, default=0.999, nargs='?')
	parser.add_argument('-decay_rate_d', type=float, default=0.9, nargs='?')
	parser.add_argument('-weight_decay_d', type=float, default=0.0, nargs='?')
	parser.add_argument('-start_decay_at', type=int, default=40000)
	parser.add_argument('-decay_method', type=str, default='exp')
	parser.add_argument('-decay_by_epoch', type=str2bool, default=True)
	parser.add_argument('-min_lr', type=float, default=1e-8)
	
	#------training
	parser.add_argument('-debug', type=str2bool, default=True)
	parser.add_argument('-train_iters', type=int, default=40000)
	parser.add_argument('-log_step', type=int, default=10)
	parser.add_argument('-drawlog_step', type=int, default=100)
	parser.add_argument('-eval_step', type=int, default=100)
	parser.add_argument('-train_from', type=str, default=None, nargs='?')
	parser.add_argument('-load_g_from', type=str, default=None, nargs='?')

	config=parser.parse_args()
	print(' '.join(sys.argv))
	
	torch.backends.cudnn.benchmark=config.benchmark
	if config.seed == -1:
		config.seed=np.random.randint(10000)
		print('set the seed as', config.seed)
	random.seed(config.seed)
	np.random.seed(config.seed)
	torch.manual_seed(config.seed)
	torch.cuda.manual_seed(config.seed)
	
	os.environ['TZ']='China'
	time.tzset()
	print(config)

	print('Start time: ', time.strftime('%X %x %Z'))
	trainer = Solver_GD(config)
	if config.save_only:
		trainer.save_model()
	elif config.labels_per_class != -1:
		trainer.train_semisup()
	else:
		trainer.train_sup()
	print('Finish time: ', time.strftime('%X %x %Z'))

