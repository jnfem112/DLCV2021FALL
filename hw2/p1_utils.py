import argparse
import os
import random as rd
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset , DataLoader
from torchvision import transforms

class Dataset(Dataset):
	def __init__(self , directory , transform):
		self.directory = directory
		self.image_name = os.listdir(directory)
		self.transform = transform

	def __len__(self):
		return len(self.image_name)

	def __getitem__(self , index):
		image = Image.open(os.path.join(self.directory , self.image_name[index]))
		return self.transform(image)

def my_argparse():
	parser = argparse.ArgumentParser()
	parser.add_argument('--train_directory' , type = str , default = 'hw2_data/face/train/')
	parser.add_argument('--test_directory' , type = str , default = 'hw2_data/face/test/')
	parser.add_argument('--output_directory' , type = str , default = 'p1_output/')
	parser.add_argument('--number_of_output' , type = int , default = 1000)
	parser.add_argument('--model' , type = str , default = 'DCGAN')
	parser.add_argument('--checkpoint' , type = str , default = 'DCGAN_generator.pth')
	parser.add_argument('--input_dim' , type = int , default = 100)
	parser.add_argument('--batch_size' , type = int , default = 64)
	parser.add_argument('--learning_rate_generator' , type = float , default = 0.0001)
	parser.add_argument('--learning_rate_discriminator' , type = float , default = 0.0001)
	parser.add_argument('--epoch' , type = int , default = 300)
	parser.add_argument('--n_critic' , type = int , default = 1)
	args = parser.parse_args()
	return args

def set_random_seed(seed):
	rd.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

def get_transform():
	return transforms.Compose([
		transforms.ToTensor() ,
		transforms.Normalize(mean = [0.5 , 0.5 , 0.5] , std = [0.5 , 0.5 , 0.5])
	])

def get_dataloader(directory , batch_size = 1024 , num_workers = 8):
	transform = get_transform()
	dataset = Dataset(directory , transform)
	dataloader = DataLoader(dataset , batch_size = batch_size , shuffle = True , num_workers = num_workers)
	return dataloader

def print_progress(epoch , total_epoch , total_data , batch_size , batch , total_batch , total_time = None , loss_generator = None , loss_discriminator = None):
	if batch < total_batch:
		data = batch * batch_size
		length = int(50 * data / total_data)
		bar = length * '=' + '>' + (49 - length) * ' '
		print('\repoch {}/{} ({}/{}) [{}]'.format(epoch , total_epoch , data , total_data , bar) , end = '')
	else:
		data = total_data
		bar = 50 * '='
		print('\repoch {}/{} ({}/{}) [{}] ({}s) generator loss : {:.8f} , discriminator loss : {:.8f}'.format(epoch , total_epoch , data , total_data , bar , total_time , loss_generator , loss_discriminator))