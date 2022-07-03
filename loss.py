#from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F

def recloss(x_rec, x_target, size_average, average_l2):
	if x_rec.dim()!=2:
		bsz = x_rec.size(0)
		x_rec = x_rec.view(bsz, -1)
		x_target = x_target.view(bsz, -1)
	reduce_func=torch.mean if average_l2 else torch.sum
	loss=reduce_func((x_rec - x_target)**2, 1)
	return loss.mean() if size_average else loss.sum()
	

def etloss(logits, size_average):
	p=F.softmax(logits, dim=1)
	logp=F.log_softmax(logits, dim=1)
	et=-torch.sum(p*logp, 1)
	return et.mean() if size_average else et.sum()


def correct_num(y, t, size_average):

	with torch.no_grad():
		acc = (torch.argmax(y, dim=1) == t).float()
		acc = acc.mean() if size_average else acc.sum()
	return acc.item()


def softmax_mse_loss(logits, target_prob, sym, size_average, average_l2):
	pred_prob = F.softmax(logits, dim=1)
	if not sym:
		target_prob = target_prob.detach()
	reduce_func = torch.mean if average_l2 else torch.sum
	loss = reduce_func((pred_prob - target_prob)**2, 1)
	return loss.mean() if size_average else loss.sum()

def kl_loss(logits, target_prob, sym, size_average):
	pred_logp = F.log_softmax(logits, dim=1)
	loss = F.kl_div(pred_logp, target_prob.detach(), reduction='batchmean' if size_average else 'sum')
	if sym:
		pred_prob = F.softmax(logits.detach(), dim=1)
		target_logp = torch.log(target_prob + 1e-10)
		loss = loss + F.kl_div(target_logp, pred_prob, reduction='batchmean' if size_average else 'sum')
	return loss



