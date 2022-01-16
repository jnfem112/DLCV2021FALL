import argparse
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset , DataLoader

class Dataset(Dataset):
	def __init__(self , data , label , transform):
		self.data = data
		self.label = label
		self.transform = transform

	def __len__(self):
		return len(self.data)

	def __getitem__(self , index):
		if self.label is not None:
			return self.transform(self.data[index]) , self.label[index]
		else:
			return self.transform(self.data[index])

# reference : https://github.com/orobix/Prototypical-Networks-for-Few-shot-Learning-PyTorch/blob/master/src/prototypical_batch_sampler.py
class Sampler(object):
	def __init__(self, labels, classes_per_it, num_samples, iterations):
		super(Sampler, self).__init__()
		self.labels = labels
		self.classes_per_it = classes_per_it
		self.sample_per_class = num_samples
		self.iterations = iterations

		self.classes, self.counts = np.unique(self.labels, return_counts=True)
		self.classes = torch.LongTensor(self.classes)
		self.idxs = range(len(self.labels))
		self.indexes = np.empty((len(self.classes), max(self.counts)), dtype=int) * np.nan
		self.indexes = torch.Tensor(self.indexes)
		self.numel_per_class = torch.zeros_like(self.classes)
		for idx, label in enumerate(self.labels):
			label_idx = np.argwhere(self.classes == label).item()
			self.indexes[label_idx, np.where(np.isnan(self.indexes[label_idx]))[0][0]] = idx
			self.numel_per_class[label_idx] += 1

	def __iter__(self):
		spc = self.sample_per_class
		cpi = self.classes_per_it

		for it in range(self.iterations):
			batch_size = spc * cpi
			batch = torch.LongTensor(batch_size)
			c_idxs = torch.randperm(len(self.classes))[:cpi]
			for i, c in enumerate(self.classes[c_idxs]):
				s = slice(i * spc, (i + 1) * spc)
				label_idx = torch.arange(len(self.classes)).long()[self.classes == c].item()
				sample_idxs = torch.randperm(self.numel_per_class[label_idx])[:spc]
				batch[s] = self.indexes[label_idx][sample_idxs]
			batch = batch[torch.randperm(len(batch))]
			yield batch

	def __len__(self):
		return self.iterations

def my_argparse():
	parser = argparse.ArgumentParser()
	parser.add_argument('--train_directory' , type = str , default = 'hw4_data/mini/train/')
	parser.add_argument('--train_annotation' , type = str , default = 'hw4_data/mini/train.csv')
	parser.add_argument('--validation_directory' , type = str , default = 'hw4_data/mini/val/')
	parser.add_argument('--validation_annotation' , type = str , default = 'hw4_data/mini/val.csv')
	parser.add_argument('--test_directory' , type = str , default = 'hw4_data/mini/val/')
	parser.add_argument('--test_annotation' , type = str , default = 'hw4_data/mini/val.csv')
	parser.add_argument('--output_file' , type = str , default = 'prediction.csv')
	parser.add_argument('--checkpoint_directory' , type = str , default = './')
	parser.add_argument('--checkpoint' , type = str , default = 'prototypical_network.pth')
	parser.add_argument('--train_way' , type = int , default = 15)
	parser.add_argument('--eval_way' , type = int , default = 5)
	parser.add_argument('--train_shot' , type = int , default = 1)
	parser.add_argument('--eval_shot' , type = int , default = 1)
	parser.add_argument('--train_query' , type = int , default = 15)
	parser.add_argument('--eval_query' , type = int , default = 75)
	parser.add_argument('--train_episode' , type = int , default = 600)
	parser.add_argument('--eval_episode' , type = int , default = 600)
	parser.add_argument('--learning_rate' , type = float , default = 0.001)
	parser.add_argument('--weight_decay' , type = float , default = 0)
	parser.add_argument('--epoch' , type = int , default = 100)
	args = parser.parse_args()
	return args

def get_transform(mode):
	if mode == 'train':
		return transforms.Compose([
			transforms.Resize((84 , 84)) , 
			transforms.RandomHorizontalFlip() ,
			transforms.RandomAffine(10 , translate = (0.1 , 0.1) , scale = (0.9 , 1.1)) ,
			transforms.ColorJitter(brightness = 0.15 , contrast = 0.15 , saturation = 0.15 , hue = 0.15) , 
			transforms.ToTensor()
		])
	else:
		return transforms.Compose([
			transforms.Resize((84 , 84)) , 
			transforms.ToTensor()
		])

def get_dataloader(data , label , args , mode):
	transform = get_transform(mode)
	dataset = Dataset(data , label , transform)
	if mode == 'train':
		sampler = Sampler(labels = label , classes_per_it = args.train_way , num_samples = args.train_shot + args.train_query , iterations = args.train_episode)
	else:
		sampler = Sampler(labels = label , classes_per_it = args.eval_way , num_samples = args.eval_shot + args.eval_query , iterations = args.eval_episode)
	dataloader = DataLoader(dataset , batch_sampler = sampler)
	return dataloader

def print_progress(epoch , total_epoch , batch , total_batch , total_time = None , loss = None , accuracy = None):
	if batch < total_batch:
		length = int(50 * batch / total_batch)
		bar = length * '=' + '>' + (49 - length) * ' '
		print('\repoch {}/{} ({}/{}) [{}]'.format(epoch , total_epoch , batch , total_batch , bar) , end = '')
	else:
		bar = 50 * '='
		print('\repoch {}/{} ({}/{}) [{}] ({}s) train loss : {:.8f} , train accuracy : {:.5f}'.format(epoch , total_epoch , batch , total_batch , bar , total_time , loss , accuracy))