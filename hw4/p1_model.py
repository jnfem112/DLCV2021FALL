import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ParametricDistance(nn.Module):
	def __init__(self):
		super(ParametricDistance , self).__init__()

		self.distance = nn.Sequential(
			nn.Linear(3200 , 256 , bias = True) , 
			nn.ReLU() , 
			nn.Linear(256 , 1 , bias = True)
		)
	
	def forward(self , x , y):
		N , D = x.shape
		M , D = y.shape
		x = x.unsqueeze(dim = 1).expand(N , M , D)
		y = y.unsqueeze(dim = 0).expand(N , M , D)
		return self.distance(torch.cat((x , y) , dim = 2)).squeeze()

class PrototypicalNetwork(nn.Module):
	def __init__(self):
		super(PrototypicalNetwork , self).__init__()

		self.convolution = nn.Sequential(
			nn.Conv2d(3 , 64 , kernel_size = 3 , stride = 1 , padding = 1) ,
			nn.BatchNorm2d(64) ,
			nn.ReLU() ,
			nn.MaxPool2d(2) ,
			nn.Conv2d(64 , 64 , kernel_size = 3 , stride = 1 , padding = 1) ,
			nn.BatchNorm2d(64) ,
			nn.ReLU() ,
			nn.MaxPool2d(2) ,
			nn.Conv2d(64 , 64 , kernel_size = 3 , stride = 1 , padding = 1) ,
			nn.BatchNorm2d(64) ,
			nn.ReLU() ,
			nn.MaxPool2d(2) ,
			nn.Conv2d(64 , 64 , kernel_size = 3 , stride = 1 , padding = 1) ,
			nn.BatchNorm2d(64) ,
			nn.ReLU() ,
			nn.MaxPool2d(2)
		)

		# self.parametric_distance = ParametricDistance()

	def forward(self , x):
		x = self.convolution(x)
		x = x.view(x.size(dim = 0) , -1)
		return x

	def Euclidean_distance(self , x , y):
		N , D = x.shape
		M , D = y.shape
		x = x.unsqueeze(dim = 1).expand(N , M , D)
		y = y.unsqueeze(dim = 0).expand(N , M , D)
		return torch.pow(x - y , 2).sum(dim = 2)

	def cosine_similarity(self , x , y):
		return x @ y.T / (x.norm(dim = 1).view(-1 , 1) + 1e-8) / (y.norm(dim = 1) + 1e-8)
	
	# reference : https://github.com/orobix/Prototypical-Networks-for-Few-shot-Learning-PyTorch/blob/master/src/prototypical_loss.py
	def prototypical_loss(self, input, target, n_support):
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		target_cpu = target.to('cpu')
		input_cpu = input.to('cpu')

		def supp_idxs(c):
			return target_cpu.eq(c).nonzero()[:n_support].squeeze(1)

		classes = torch.unique(target_cpu)
		n_classes = len(classes)
		n_query = target_cpu.eq(classes[0].item()).sum().item() - n_support
		support_idxs = list(map(supp_idxs, classes))
		prototypes = torch.stack([input_cpu[idx_list].mean(0) for idx_list in support_idxs])
		query_idxs = torch.stack(list(map(lambda c: target_cpu.eq(c).nonzero()[n_support:], classes))).view(-1)
		query_samples = input.to('cpu')[query_idxs]
		dists = self.Euclidean_distance(query_samples, prototypes)
		# dists = self.cosine_similarity(query_samples, prototypes)
		# dists = self.parametric_distance(query_samples.to(device) , prototypes.to(device)).to('cpu')
		log_p_y = F.log_softmax(-dists, dim=1).view(n_classes, n_query, -1)
		target_inds = torch.arange(0, n_classes)
		target_inds = target_inds.view(n_classes, 1, 1)
		target_inds = target_inds.expand(n_classes, n_query, 1).long()
		loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
		_, y_hat = log_p_y.max(2)
		acc_val = y_hat.eq(target_inds.squeeze()).float().mean()
		return loss_val,  acc_val