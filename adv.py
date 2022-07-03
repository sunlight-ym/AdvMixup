import torch
import torch.nn.functional as F

def l2_normalize(x):
	shape = x.size()
	x = x.view(shape[0], -1)
	x = F.normalize(x, dim=1)
	return x.view(shape)

def clip_p(p, max_norm, eps):
	if max_norm:
		return torch.clamp(p, -eps, eps)
	else:
		shape = p.size()
		p = p.view(shape[0], -1)
		norm = torch.clamp(p.norm(2, 1, keepdim=True), min=1e-6)
		factor = torch.clamp(eps / norm, max=1.0)
		p = p * factor
		return p.view(shape)
	

class AEG(object):
	def __init__(self, method, epsilon, max_norm, step_epsilon=None, xi=None, n_iter=None, l2_dist=None):
		self.method = method
		self.epsilon = epsilon
		self.max_norm = max_norm
		self.step_epsilon = step_epsilon
		self.xi=xi
		self.n_iter=n_iter
		self.l2_dist = l2_dist

	def __call__(self, x, t, source_model):
		if self.method=='fgsm':
			x_adv=self.fgsm(x, t, source_model, self.max_norm, self.epsilon)
		elif self.method =='ifgsm':
			x_adv = self.iter_fgsm(x, t, source_model, self.n_iter, self.max_norm, self.epsilon, self.step_epsilon)
		elif self.method=='vat':
			x_adv=self.vat(x, source_model, self.n_iter, self.xi, self.l2_dist, self.epsilon)
		else:
			raise ValueError('unsupported adversarial example crafting method!')

		return x_adv
		
	def fgsm(self, x, t, model, max_norm, epsilon):
		
		x = x.detach().requires_grad_()
		pred=model(x)
		loss=F.cross_entropy(pred, t, reduction='sum')
		p = torch.autograd.grad(loss, x)[0]
		if max_norm:
			p=p.sign()
		else:
			p=l2_normalize(p)
		
		x_adv = x.detach() + epsilon * p
		return x_adv

	def iter_fgsm(self, x, t, model, n_iter, max_norm, epsilon, step_epsilon):
		x_adv = x
		for _ in range(n_iter):
			x_adv = self.fgsm(x_adv, t, model, max_norm, step_epsilon)
			p = x_adv - x
			p = clip_p(p, max_norm, epsilon)
			x_adv = x + p
		return x_adv


	def vat(self, x, model, n_iter, xi, l2_dist, epsilon):
		with torch.no_grad():
			target = model(x)
			target = F.softmax(target, dim=1)
		d = torch.randn_like(x)
		with torch.enable_grad():
			for _ in range(n_iter):
				d = xi * l2_normalize(d)
				d.requires_grad_()
				logit = model(x + d)
				if l2_dist:
					p = F.softmax(logit, dim=1)
					loss = F.mse_loss(p, target, reduction = 'sum')
				else:
					logp = F.log_softmax(logit, dim=1)
					loss = F.kl_div(logp, target, reduction='batchmean')
				d = torch.autograd.grad(loss, d)[0]

		p = epsilon * l2_normalize(d)
		return x + p





