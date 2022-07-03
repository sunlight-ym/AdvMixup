import argparse
import time
import os
import sys
import random
import numpy as np
import torch
from data_loader import load_data
from utils import *
from adv import AEG


def str2bool(v):
	return v.lower() in ('true')

parser=argparse.ArgumentParser()
parser.add_argument('v', type=str)
parser.add_argument('-seed', type=int, default=1)
parser.add_argument('-target_model_path', type=str, default=None)
parser.add_argument('-source_model_path', type=str, default=None, nargs='?')
parser.add_argument('-gpu', type=int, default=0)
parser.add_argument('-dataset', type=str, default='cifar')
parser.add_argument('-data_path', type=str, default='../datasets')
parser.add_argument('-batch_size', type=int, default=100)

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

random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed(config.seed)



def eval(data_loader, target_model, source_model, adv_generator):
	start_time=time.time()
	n_total=len(data_loader.dataset)
	target_dis = target_model.dis_ema if target_model.use_ema else target_model.dis
	source_dis = source_model.dis_ema if source_model.use_ema else source_model.dis
	acc = 0
	for x, t in data_loader:
		x, t = to_device(device, (x, t))
		x = source_model.preprocess(x)
		x_adv = adv_generator(x, t, source_dis)
		with torch.no_grad():
			pred = target_dis(x_adv)
			acc += (pred.argmax(1) == t).float().sum().item()

	acc = acc / n_total
	print('Acc: {:.4f}, evaluation takes {:6.0f} seconds'.format(acc, time.time()-start_time))




if __name__=='__main__':

	data_path = os.path.join(config.data_path, config.dataset, 'data')
	test_loader = load_data(config.dataset, data_path, 'd', False, 
		config.batch_size, config.batch_size, config.batch_size, True, 0, 
		-1, 0, examples_per_class=0)['test']

	target_model = torch.load(config.target_model_path, map_location=device)['model']
	frozen_model(target_model)
	target_model.eval()

	if config.source_model_path is not None:
		source_model = torch.load(config.source_model_path, map_location=device)['model']
		frozen_model(source_model)
		source_model.eval()
	else:
		source_model = target_model

	adv_generator = AEG(config.adv_mode, config.epsilon, config.max_norm, config.step_epsilon, config.xi, config.n_iter, config.l2_dist)
	eval(test_loader, target_model, source_model, adv_generator)
		
	

	



