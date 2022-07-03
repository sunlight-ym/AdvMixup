#from __future__ import division
import math
import time
import torch
from torch import optim
from data_loader import load_data
from loss import *
from utils import *
from model import GD_model
import os
from collections import OrderedDict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class Solver_GD(object):
	"""docstring for Solver"""
	def __init__(self, config):
		super(Solver_GD, self).__init__()
		self.device = torch.device('cuda', config.gpu)
		

		self.use_ema = config.use_ema
		self.ema_decay = config.ema_decay

		self.nll_weight=config.nll_weight		
		self.adv_weight=config.adv_weight
		self.mixup_weight=config.mixup_weight
		self.entropy_weight=config.entropy_weight
		self.l2_dist = config.l2_dist
		self.average_l2=config.average_l2
		self.sym = config.sym
		self.sym_start_at = config.sym_start_at
		self.cur_sym = False

		self.temperature = config.temperature
		self.temp_rampdown_iters_end = config.temp_rampdown_iters_end
		self.temp_rampdown_iters_start = config.temp_rampdown_iters_start
		self.temp_final = config.temp_final
		# if self.sym:
		# 	assert not self.use_ema, 'using ema model cannot support symmetric update'
		
		self.weight_rampup_iters_end = config.weight_rampup_iters_end
		self.weight_rampup_iters_start = config.weight_rampup_iters_start
		
		self.lr_d=config.lr_d
		self.lr_rampup_iters_end=config.lr_rampup_iters_end
		self.lr_rampup_iters_start=config.lr_rampup_iters_start
		self.start_decay_at=config.start_decay_at
		self.decay_method=config.decay_method
		self.decay_rate_d=config.decay_rate_d
		self.min_lr=config.min_lr

		self.debug=config.debug
		self.train_iters=config.train_iters
		data_path = os.path.join(config.data_path, config.dataset, 'data')
		# transmode = 'g' if self.g_only else ('gd' if self.use_g else 'd')
		loaded_datasets = load_data(config.dataset, data_path, 'd', config.data_aug, 
			config.batch_size, config.ul_batch_size, config.eval_batch_size, config.disjoint, config.workers, 
			config.labels_per_class, config.valid_labels_per_class, config.cifar100_zca, examples_per_class=0)

		self.label_train_loader=loaded_datasets['label']
		self.unlabel_train_loader=loaded_datasets['unlabel']
		self.valid_loader=loaded_datasets['valid']
		self.test_loader=loaded_datasets['test']
		config.num_classes = loaded_datasets['nc']
		config.input_size = loaded_datasets['input_size']
		config.zca_mean, config.zca_components = loaded_datasets['zca']
		print('label dataset size:', len(self.label_train_loader.sampler))
		print('unlabel dataset size:', len(self.unlabel_train_loader.sampler) if self.unlabel_train_loader is not None else 0)
		print('valid dataset size:', len(self.valid_loader.sampler) if self.valid_loader is not None else 0)
		print('test dataset size:', len(self.test_loader.dataset))
		print('number of classes:', config.num_classes)
		print('input size:', config.input_size)

		if config.decay_by_epoch:
			if config.labels_per_class==-1:
				self.decay_step=len(self.label_train_loader.dataset)/config.batch_size 
			else:
				self.decay_step=max(len(self.label_train_loader.dataset)/config.batch_size, len(self.unlabel_train_loader.dataset)/config.ul_batch_size)
			self.decay_step=int(self.decay_step)
		else:
			self.decay_step = 1
		print('seting decay_step as', self.decay_step)

		self.combine=config.disjoint
		self.log_step=config.log_step
		self.drawlog_step=config.drawlog_step
		self.eval_step=config.eval_step
		
		self.drawlog_path=os.path.join(config.save_dir, config.dataset, 'curve', config.v)
		makedirs(self.drawlog_path)
		self.model_path=os.path.join(config.save_dir, config.dataset, 'model', config.v)
		makedirs(self.model_path)
		

		self.model=GD_model(config)
		print('number of parameters in total:', sum([p.nelement() for p in self.model.parameters()]))
		self.model.to(self.device)
		self.build_optimizer(config)
		self.keys, self.datasets, self.loss_flags = self.init_keys()
		if config.train_from is not None:
			check_point=torch.load(config.train_from, map_location=lambda storage, loc: storage)
			self.model.load_state_dict(check_point['model_state'])
			self.optimizer_d.load_state_dict(check_point['optimizer_d_state'])
			self.start_step = check_point['step'] + 1
			self.best_record = check_point['best_record']
			self.logs = check_point['logs']
			del check_point
		else:
			self.start_step=1
			record_min = 0
			self.best_record = {'test': (record_min, 0)}
			if self.valid_loader is not None:
				self.best_record['valid'] = (record_min, 0)
			# if self.use_ema:
			# 	self.best_record['test_avg'] = (record_min, 0)
			self.logs=self.init_log()
		
		self.cur_step=self.start_step
	
	def build_optimizer(self, config):
		if config.optim_method=='adam':
			self.optimizer_d=optim.Adam(self.model.dis.parameters(), config.lr_d, (config.beta1_d, config.beta2_d), weight_decay=config.weight_decay_d)
		elif config.optim_method=='rmsprop':
			self.optimizer_d=optim.RMSprop(self.model.dis.parameters(), config.lr_d, weight_decay=config.weight_decay_d)
		elif config.optim_method=='sgd':
			self.optimizer_d=optim.SGD(self.model.dis.parameters(), config.lr_d, momentum=0.9, weight_decay=config.weight_decay_d)
		else:
			raise ValueError('unsurported optimization method!')

	def save_states(self, prefix = ''):
		check_point = {
			'step': self.cur_step,
			'model_state': self.model.state_dict(),
			'optimizer_d_state': self.optimizer_d.state_dict(),
			'best_record': self.best_record,
			'logs': self.logs
		}
		
		filename = os.path.join(self.model_path, '{}model-{}'.format(prefix, self.cur_step))
		torch.save(check_point, filename)
	def save_model(self):
		check_point = {
			'model': self.model
		}
		filename = os.path.join(self.model_path, 'full-model-{}'.format(self.cur_step))
		torch.save(check_point, filename)
		print('model is saved at {}'.format(filename))
		
	def update_ema_variables(self):
		# Use the true average until the exponential average is more correct
		alpha = min(1 - 1 / (self.cur_step + 1), self.ema_decay)
		for ema_param, param in zip(self.model.dis_ema.parameters(), self.model.dis.parameters()):
			ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
	def init_keys(self):
		keys=[]
		#here comes the optional losses and initialize them as False
		loss_flags={}
		for k in ['acc', 'acc_ul', 'acc_ul_adv', 'acc_ul_mix', 'D_loss', 'nll_loss', 'adv_loss', 'mix_loss', 'ent_loss']:
			loss_flags[k]=False
		active_losses=[]
		
		active_losses.append('acc')
		active_losses.append('D_loss')
		active_losses.append('nll_loss')
		if self.unlabel_train_loader is not None and (self.mixup_weight>0 or self.adv_weight>0 or self.entropy_weight>0):
			active_losses.append('acc_ul')
		if self.mixup_weight>0:
			active_losses.append('mix_loss')
			active_losses.append('acc_ul_mix')
		if self.adv_weight>0:
			active_losses.append('adv_loss')
			active_losses.append('acc_ul_adv')
		if self.entropy_weight>0:
			active_losses.append('ent_loss')
		

		for k in active_losses:
			keys.append(k)
			loss_flags[k]=True

		datasets=['train']
		if self.valid_loader is not None:
			datasets.append('valid')
		

		return keys, datasets, loss_flags
	
	def init_log(self):
		test_iters=self.train_iters//self.eval_step
		def _init_log(log_length):
			empty_log=OrderedDict()
			for k in self.keys:
				empty_log[k]=np.zeros(log_length)
			return empty_log
	
		log_set=OrderedDict()
		for d in self.datasets:
			log_set[d]=_init_log(self.train_iters if d=='train' else test_iters)
		return log_set

	def update_log(self, dataset, result):
		#the result here can just be an ordinary dict
		cur_step=self.cur_step if dataset=='train' else self.cur_step//self.eval_step
		for key in result:
			self.logs[dataset][key][cur_step-1]=result[key]

	def draw_log(self):
		log_num=len(self.keys)
		log_steps=self.cur_step
		log_eval_steps=log_steps//self.eval_step
		color_sets=['b','k','r','g'][:len(self.datasets)]
		train_ind=np.arange(1, log_steps+1)
		eval_ind=np.arange(self.eval_step, log_steps+1, self.eval_step)
		disp_interval=max(log_steps//10, 1)
		xtl=np.arange(disp_interval, log_steps+1, disp_interval)
		fig_num=math.ceil(len(self.keys)/6)
		for r in range(fig_num):
			sub_fname=self.drawlog_path+'/'+str(r)+'.pdf'
			keys=self.keys[r*6:(r+1)*6] if r!=fig_num-1 else self.keys[r*6:]
			sub_fig_num=len(keys)
			handles, handle_labels=[], []
			f, axes=plt.subplots(sub_fig_num, sharex=True, squeeze=False)
			for i, k in enumerate(keys):
				for j, dataset in enumerate(self.datasets):
					ind = train_ind if j==0 else eval_ind
					cut_at = log_steps if j==0 else log_eval_steps
					if cut_at>0:
						h,=axes[i,0].plot(ind, self.logs[dataset][k][:cut_at], color_sets[j], lw=0.3)
						if i==sub_fig_num-1:
							handles.append(h)
							handle_labels.append(dataset)
				axes[i,0].grid(True)
				axes[i,0].set_ylabel(k, rotation=0)
				axes[i,0].yaxis.set_label_position("right")
				if i==sub_fig_num-1:
					axes[i,0].set_xticks(xtl)
			f.legend(handles, handle_labels, loc='upper right')
			f.subplots_adjust(hspace=0)
			f.savefig(sub_fname)
			plt.close(f)
	
	def compute_loss(self, result, size_average):
		
		details={}
		if self.loss_flags['acc']:
			details['acc']=correct_num(result['x_pred'], result['t'], size_average)
		if self.loss_flags['acc_ul']:
			details['acc_ul']=correct_num(result['pred_target'] if 'pred_target' in result else result['ent'], result['ul_t'], size_average) if self.model.training else 0
		if self.loss_flags['acc_ul_adv']:
			details['acc_ul_adv']=correct_num(result['adv'], result['ul_t_adv'], size_average)
		if self.loss_flags['acc_ul_mix']:
			details['acc_ul_mix']=result['mix'][3] * correct_num(result['mix'][0], result['ul_t_mix'], size_average) + (1 - result['mix'][3]) * correct_num(result['mix'][0], result['mix'][2], size_average)
		
		up_weight = rampup(self.cur_step, self.weight_rampup_iters_end, self.weight_rampup_iters_start, decay_step=self.decay_step)
		D_loss = 0

		if self.loss_flags['nll_loss']:
			nll_loss=F.cross_entropy(result['x_pred'], result['t'], reduction='mean' if size_average else 'sum')
			details['nll_loss']=nll_loss.item()
			D_loss = D_loss + self.nll_weight * nll_loss

		if self.loss_flags['adv_loss']:
			if self.l2_dist:
				adv_loss = softmax_mse_loss(result['adv'], result['pred_target_adv'], self.cur_sym, size_average, self.average_l2)
			else:
				adv_loss = kl_loss(result['adv'], result['pred_target_adv'], self.cur_sym, size_average)
			details['adv_loss'] = adv_loss.item()
			if self.adv_weight > 0:
				D_loss = D_loss + self.adv_weight * up_weight * adv_loss
			
		if self.loss_flags['mix_loss']:
			if self.l2_dist:
				mix_loss = softmax_mse_loss(result['mix'][0], result['mix'][1], self.cur_sym, size_average, self.average_l2)
			else:
				mix_loss = kl_loss(result['mix'][0], result['mix'][1], self.cur_sym, size_average)
			details['mix_loss'] = mix_loss.item()
			if self.mixup_weight > 0:
				D_loss = D_loss + self.mixup_weight * up_weight * mix_loss
			

		if self.loss_flags['ent_loss']:
			ent_loss=etloss(result['ent'], size_average)
			details['ent_loss']=ent_loss.item()
			D_loss = D_loss + self.entropy_weight * ent_loss
			
		


		
		details['D_loss'] = D_loss.item()
			

		if self.model.training:
			
			self.update_d(D_loss)
			
			if self.use_ema:
				self.update_ema_variables()
		
		return details
	
	
	def update_d(self, loss, g_out=None):
		self.optimizer_d.zero_grad()
		if g_out is not None:
			g_out.detach_()
		loss.backward()
		self.optimizer_d.step()

	def update_lr(self, optimizer, init_lr, decay_rate):
		step=self.cur_step
		old_lr=optimizer.param_groups[0]['lr']
		if old_lr>self.min_lr:
			if self.decay_method=='exp' and (step-self.start_decay_at)%self.decay_step==0:
				new_lr=old_lr*decay_rate
			elif self.decay_method=='linear' and (step-self.start_decay_at)%self.decay_step==0:
				new_lr=init_lr*((self.train_iters-step)/(self.train_iters-self.start_decay_at))
			elif self.decay_method == 'cos' and (step-self.start_decay_at)%self.decay_step==0:
				new_lr = init_lr * cosine_rampdown(step-self.start_decay_at, self.train_iters-self.start_decay_at)
			else:
				return
			new_lr=max(new_lr, self.min_lr)
			if new_lr!=old_lr:
				optimizer.param_groups[0]['lr']=new_lr
				if self.debug:
					print('update the learning rate to', optimizer.param_groups[0]['lr'])
	def rampup_lr(self, optimizer, init_lr):
		new_lr=init_lr*rampup(self.cur_step, self.lr_rampup_iters_end, self.lr_rampup_iters_start, decay_step=self.decay_step)
		new_lr=max(new_lr, self.min_lr)
		optimizer.param_groups[0]['lr']=new_lr

	def print_msg(self, details, start_time=None):
		s='step [%d/%d]'%(self.cur_step, self.train_iters)
		for i,k in enumerate(details):
			s+=' %s: %.4f' % (k, details[k])
			if i==5:
				s+='\n'
		print(s)
		if start_time is not None:
			print('%6.0f s elapsed' % (time.time()-start_time))
	def print_record(self, record, dataset_name, attr_name):
		print('current best {} {}: {:.4f} at step {}'.format(dataset_name, attr_name, record[0], record[1]))

	def update_temp(self):
		if self.temp_rampdown_iters_start is not None:
			if self.cur_step <= self.temp_rampdown_iters_start:
				return self.temperature
			elif self.cur_step < self.temp_rampdown_iters_end:
				ratio = (self.temp_rampdown_iters_end - self.cur_step) / (self.temp_rampdown_iters_end - self.temp_rampdown_iters_start)
				return self.temp_final + (self.temperature - self.temp_final) * ratio
			else:
				return self.temp_final
		else:
			return self.temperature
	
	def train_semisup(self):
		label_data_iter=iter(self.label_train_loader)
		unlabel_data_iter=iter(self.unlabel_train_loader)
		label_data_batches=len(label_data_iter)
		unlabel_data_batches=len(unlabel_data_iter)
		self.model.train()
		start_time=time.time()


		for step in range(self.start_step, self.train_iters+1):
			self.cur_step=step
			if step<self.lr_rampup_iters_end:
				self.rampup_lr(self.optimizer_d, self.lr_d)
				
			self.cur_sym = (self.sym and self.cur_step > self.sym_start_at)
			cur_temp = self.update_temp()
					
			x, t = next(label_data_iter)
			u, ul_t = next(unlabel_data_iter)
			x, t, u, ul_t = to_device(self.device, (x, t, u, ul_t))
			
			result = self.model.train_d(x, u, t, ul_t, cur_temp, self.combine, self.cur_sym, self.loss_flags['mix_loss'], self.adv_weight > 0, self.loss_flags['ent_loss'])

			details=self.compute_loss(result, True)
			
			if step%label_data_batches==0:
				label_data_iter=iter(self.label_train_loader)
			if step%unlabel_data_batches==0:
				unlabel_data_iter=iter(self.unlabel_train_loader)
			self.post_training(details, start_time)
			
	def train_sup(self):
		label_data_iter=iter(self.label_train_loader)
		label_data_batches=len(label_data_iter)
		self.model.train()
		start_time=time.time()


		for step in range(self.start_step, self.train_iters+1):
			self.cur_step=step
			if step<self.lr_rampup_iters_end:
				self.rampup_lr(self.optimizer_d, self.lr_d)
				
			self.cur_sym = (self.sym and self.cur_step > self.sym_start_at)
			cur_temp = self.update_temp()
					
			x, t = next(label_data_iter)
			x, t = to_device(self.device, (x, t))
			
			result = self.model.train_d(x, x, t, t, cur_temp, False, self.cur_sym, self.loss_flags['mix_loss'], self.adv_weight > 0, self.loss_flags['ent_loss'], sup=True)

			details=self.compute_loss(result, True)
			
			if step%label_data_batches==0:
				label_data_iter=iter(self.label_train_loader)
			self.post_training(details, start_time)

	
			
	def post_training(self, details, start_time):
		step=self.cur_step
		self.update_log('train', details)
		if step%self.log_step==0:
			self.print_msg(details, start_time)
			
		if step%self.eval_step==0:
			
			#evaluate on valid and test set
			if self.valid_loader is not None:
				valid_acc=self.eval(self.valid_loader, 'valid', self.use_ema)
			test_acc=self.eval(self.test_loader, 'test', self.use_ema)
			
			save_flag=False
			attr_name = 'acc'
			if self.valid_loader is not None:
				if valid_acc > self.best_record['valid'][0]:
					self.best_record['valid'] = (valid_acc, step)
				self.print_record(self.best_record['valid'], 'valid', attr_name)
			if test_acc>self.best_record['test'][0]:
				self.best_record['test'] = (test_acc, step)
				save_flag=True
			self.print_record(self.best_record['test'], 'test', attr_name)
			
			
			if save_flag:
				self.save_states()
			self.save_states('latest-')
			if step!=self.eval_step:
				os.remove(os.path.join(self.model_path, 'latest-model-%d'%(step-self.eval_step)))

		if step>self.start_decay_at:
			self.update_lr(self.optimizer_d, self.lr_d, self.decay_rate_d)
				
		if step%self.drawlog_step==0:
			#draw logs
			self.draw_log()


	
	
	def eval(self, data_loader, dataset_name, ema_model):
		data_iter=iter(data_loader)
		n_total=len(data_loader.dataset) if dataset_name == 'test' else len(data_loader.sampler)
		self.model.eval()
		acc = 0

		dis_model = self.model.dis_ema if ema_model else self.model.dis
		with torch.no_grad():
			for i, (x, t) in enumerate(data_iter):
				x, t = to_device(self.device, (x, t))
				pred = dis_model(self.model.preprocess(x))
				acc += correct_num(pred, t, False)
		acc /= n_total
		print('%s performance: %.4f'%(dataset_name, acc))
		self.model.train()
		return acc

