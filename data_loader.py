#from __future__ import division
import torch
from torchvision import datasets
from torchvision import transforms
import numpy as np
from scipy import linalg
from functools import reduce
from operator import __or__
from torch.utils.data.sampler import SubsetRandomSampler

# transform=transforms.Compose([
# 				transforms.ToTensor(),
# 				transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
# 				])
def load_data(dataset, path, transmode, data_aug, batch_size, ul_batch_size, eval_batch_size, disjoint, workers, labels_per_class, valid_labels_per_class, cifar100_zca, examples_per_class=0):
	## copied from GibbsNet_pytorch/load.py    
		
	if dataset == 'cifar10':
		mean = [x / 255 for x in [125.3, 123.0, 113.9]]
		std = [x / 255 for x in [63.0, 62.1, 66.7]]
		input_size = (3, 32, 32)
	elif dataset == 'cifar100':
		mean = [x / 255 for x in [129.3, 124.1, 112.4]]
		std = [x / 255 for x in [68.2, 65.4, 70.4]]
		input_size = (3, 32, 32)
	elif dataset == 'svhn':
		mean = [x / 255 for x in [127.5, 127.5, 127.5]]
		std = [x / 255 for x in [127.5, 127.5, 127.5]]
		input_size = (3, 32, 32)
	elif dataset == 'mnist':
		mean = (0.1307,)
		std = (0.3081,)
		input_size = (1, 24, 24) if data_aug else (1, 28, 28)
	else:
		assert False, "Unknow dataset : {}".format(dataset)

	if dataset == 'mnist':
		zero_mean, one_std = (0.5,), (0.5,)
	else:
		zero_mean, one_std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
	
	if transmode == 'g':
		normalize_tf = transforms.Normalize(zero_mean, one_std)
	elif transmode == 'gd':
		normalize_tf = parallel_transform(transforms.Normalize(mean, std), transforms.Normalize(zero_mean, one_std))
	else:
		normalize_tf = transforms.Normalize(mean, std)

	typical_transform = transforms.Compose([transforms.ToTensor(), normalize_tf])
	
	if data_aug==True:
		print ('data aug')
		if dataset == 'svhn':
			train_transform = transforms.Compose([transforms.RandomCrop(32, padding=2), transforms.ToTensor(),
											  normalize_tf])
			test_transform = typical_transform
		elif dataset == 'mnist':
			hw_size = 24
			train_transform = transforms.Compose([transforms.RandomCrop(hw_size),                
													transforms.ToTensor(),
													normalize_tf])
			test_transform = transforms.Compose([transforms.CenterCrop(hw_size),                       
												transforms.ToTensor(),
												normalize_tf])
		else:    
			train_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
												  transforms.RandomCrop(32, padding=2),
												  transforms.ToTensor(),
												  normalize_tf])
			test_transform = typical_transform
	else:
		print ('no data aug')
		train_transform = typical_transform
		test_transform = typical_transform

	if dataset == 'cifar10':
		train_data = datasets.CIFAR10(path, train=True, transform=train_transform, download=True)
		test_data = datasets.CIFAR10(path, train=False, transform=test_transform, download=True)
		num_classes = 10
	elif dataset == 'cifar100':
		train_data = datasets.CIFAR100(path, train=True, transform=train_transform, download=True)
		test_data = datasets.CIFAR100(path, train=False, transform=test_transform, download=True)
		num_classes = 100
	elif dataset == 'svhn':
		train_data = datasets.SVHN(path, split='train', transform=train_transform, download=True)
		test_data = datasets.SVHN(path, split='test', transform=test_transform, download=True)
		num_classes = 10
	elif dataset == 'mnist':
		train_data = datasets.MNIST(path, train=True, transform=train_transform, download=True)
		test_data = datasets.MNIST(path, train=False, transform=test_transform, download=True)
		num_classes = 10
	#print ('svhn', train_data.labels.shape)
	elif dataset == 'stl10':
		train_data = datasets.STL10(path, split='train', transform=train_transform, download=True)
		test_data = datasets.STL10(path, split='test', transform=test_transform, download=True)
		num_classes = 10
	elif dataset == 'imagenet':
		assert False, 'Do not finish imagenet code'
	else:
		assert False, 'Do not support dataset : {}'.format(dataset)

		
	n_labels = num_classes
	
	def get_sampler(labels, n=None, n_valid= None, n_ex=None):
		# Only choose digits in n_labels
		# n = number of labels per class for training
		# n_val = number of lables per class for validation
		#print type(labels)
		#print (n_valid)
		(indices,) = np.where(reduce(__or__, [labels == i for i in np.arange(n_labels)]))
		# Ensure uniform distribution of labels
		np.random.shuffle(indices)
		if n_valid > 0:
			indices_valid = np.hstack([list(filter(lambda idx: labels[idx] == i, indices))[:n_valid] for i in range(n_labels)])
		if n!=-1:
			indices_train = np.hstack([list(filter(lambda idx: labels[idx] == i, indices))[n_valid:n_valid+n] for i in range(n_labels)])
			indices_unlabelled = np.hstack([list(filter(lambda idx: labels[idx] == i, indices))[n_valid+(n if disjoint else 0):] for i in range(n_labels)])
		else:
			indices_train = np.hstack([list(filter(lambda idx: labels[idx] == i, indices))[n_valid:] for i in range(n_labels)])
		if n_ex > 0:
			indices_examples = np.hstack([list(filter(lambda idx: labels[idx] == i, indices))[-n_ex:] for i in range(n_labels)])
		#print (indices_train.shape)
		#print (indices_valid.shape)
		#print (indices_unlabelled.shape)
		indices_train = torch.from_numpy(indices_train)
		sampler_train = SubsetRandomSampler(indices_train)

		if n_valid > 0:
			indices_valid = torch.from_numpy(indices_valid)
			sampler_valid = SubsetRandomSampler(indices_valid)
		else:
			sampler_valid = None
		
		if n!=-1:
			indices_unlabelled = torch.from_numpy(indices_unlabelled)
			sampler_unlabelled = SubsetRandomSampler(indices_unlabelled)
		else:
			sampler_unlabelled = None
		
		if n_ex > 0:
			indices_examples = torch.from_numpy(indices_examples)
			sampler_examples = SubsetSequentialSampler(indices_examples)
		else:
			sampler_examples = None
		return sampler_train, sampler_valid, sampler_unlabelled, sampler_examples
	
	#print type(train_data.train_labels)
	
	# Dataloaders for MNIST
	if dataset == 'svhn':
		train_sampler, valid_sampler, unlabelled_sampler, ex_sampler = get_sampler(train_data.labels, labels_per_class, valid_labels_per_class, examples_per_class)
	elif dataset == 'mnist':
		train_sampler, valid_sampler, unlabelled_sampler, ex_sampler = get_sampler(train_data.train_labels.numpy(), labels_per_class, valid_labels_per_class, examples_per_class)
	else: 
		train_sampler, valid_sampler, unlabelled_sampler, ex_sampler = get_sampler(train_data.train_labels, labels_per_class, valid_labels_per_class, examples_per_class)
	
	labelled = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler = train_sampler,  num_workers=workers, pin_memory=True)
	validation = torch.utils.data.DataLoader(train_data, batch_size=eval_batch_size, sampler = valid_sampler,  num_workers=workers, pin_memory=True) if valid_labels_per_class > 0 else None
	unlabelled = torch.utils.data.DataLoader(train_data, batch_size=ul_batch_size, sampler = unlabelled_sampler,  num_workers=workers, pin_memory=True) if labels_per_class != -1 else None
	test = torch.utils.data.DataLoader(test_data, batch_size=eval_batch_size, shuffle=False, num_workers=workers, pin_memory=True)

	examples = torch.utils.data.DataLoader(train_data, batch_size=examples_per_class*num_classes, sampler = ex_sampler,  num_workers=workers, pin_memory=True) if examples_per_class > 0 else None

	result = {'input_size':input_size, 'label':labelled, 'valid':validation, 'unlabel':unlabelled, 'test':test, 'ex':examples, 'nc':num_classes}
	if (dataset == 'cifar10' or (dataset == 'cifar100' and cifar100_zca)) and transmode != 'g':
		data_zca = torch.utils.data.DataLoader(train_data, batch_size=len(train_data), num_workers=workers, pin_memory=True)
		data_zca, _ = iter(data_zca).next()
		if transmode == 'gd':
			data_zca = data_zca[0]
		data_zca = data_zca.view(data_zca.size(0), -1).numpy()
		result['zca'] = ZCA(data_zca)
	else:
		result['zca'] = (None, None)
	return result


class SubsetSequentialSampler(torch.utils.data.Sampler):
	"""Samples elements randomly from a given list of indices, without replacement.

	Arguments:
		indices (sequence): a sequence of indices
	"""

	def __init__(self, indices):
		self.indices = indices

	def __iter__(self):
		return (self.indices[i] for i in range(len(self.indices)))

	def __len__(self):
		return len(self.indices)

def parallel_transform(t1, t2):
	def pt(x):
		return t1(x), t2(x)
	return pt

def ZCA(data, reg=1e-6):
	mean=np.mean(data, axis=0)
	mdata=data-mean
	sigma=np.dot(mdata.T, mdata)/mdata.shape[0]
	U, S, _=linalg.svd(sigma)
	components=np.dot(np.dot(U, np.diag(1./np.sqrt(S)+reg)), U.T)
	return mean, components.T