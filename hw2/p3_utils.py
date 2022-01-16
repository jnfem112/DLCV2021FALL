import argparse
import os
import random as rd
import pandas as pd
from PIL import Image
from skimage import filters
import numpy as np
import torch
from torch.utils.data import Dataset , DataLoader
from torchvision import transforms

class Dataset(Dataset):
	def __init__(self , directory , image_name , label , transform , grayscale = False):
		self.directory = directory
		self.image_name = image_name
		self.label = label
		self.transform = transform
		self.grayscale = grayscale

	def __len__(self):
		return len(self.image_name)

	def __getitem__(self , index):
		image = Image.open(os.path.join(self.directory , self.image_name[index]))
		image = image.convert('L') if self.grayscale else image.convert('RGB')
		if self.label != None:
			return self.transform(image) , self.label[index]
		else:
			return self.transform(image)

def my_argparse():
	parser = argparse.ArgumentParser()
	parser.add_argument('--source' , type = str , default = '')
	parser.add_argument('--target' , type = str , default = 'mnistm')
	parser.add_argument('--source_directory' , type = str , default = 'hw2_data/digits/svhn/train/')
	parser.add_argument('--source_annotation' , type = str , default = 'hw2_data/digits/svhn/train.csv')
	parser.add_argument('--target_directory' , type = str , default = 'hw2_data/digits/mnistm/train/')
	parser.add_argument('--target_annotation' , type = str , default = 'hw2_data/digits/mnistm/train.csv')
	parser.add_argument('--test_directory' , type = str , default = 'hw2_data/digits/mnistm/test/')
	parser.add_argument('--test_annotation' , type = str , default = 'hw2_data/digits/mnistm/test.csv')
	parser.add_argument('--output_file' , type = str , default = 'mnist.csv')
	parser.add_argument('--model' , type = str , default = 'DaNN')
	parser.add_argument('--batch_size' , type = int , default = 64)
	parser.add_argument('--learning_rate' , type = float , default = 0.0001)
	parser.add_argument('--weight_decay' , type = float , default = 0)
	parser.add_argument('--lambd' , type = float , default = 0.1)
	parser.add_argument('--epoch' , type = int , default = 100)
	args = parser.parse_args()

	if not args.source:
		if args.target == 'mnistm':
			args.source = 'svhn'
		elif args.target == 'usps':
			args.source = 'mnistm'
		elif args.target == 'svhn':
			args.source = 'usps'

	return args

def otsu(image):
	image = np.array(image)
	threshold = filters.threshold_otsu(image)
	image[image <= threshold] = 0
	image[image > threshold] = 255
	image = Image.fromarray(image)
	return image

def get_transform(mode , thresholding = False):
	if mode == 'train':
		if thresholding:
			return transforms.Compose([
				transforms.RandomAffine(10 , translate = (0.1 , 0.1) , scale = (0.9 , 1.1)) ,
				transforms.Grayscale() ,
				transforms.Lambda(lambda x : otsu(x)) , 
				transforms.ToTensor()
			])
		else:
			return transforms.Compose([
				transforms.RandomAffine(10 , translate = (0.1 , 0.1) , scale = (0.9 , 1.1)) ,
				transforms.ColorJitter(brightness = 0.15 , contrast = 0.15 , saturation = 0.15 , hue = 0.15) , 
				transforms.ToTensor()
			])
	else:
		if thresholding:
			return transforms.Compose([
				transforms.Grayscale() ,
				transforms.Lambda(lambda x : otsu(x)) , 
				transforms.ToTensor()
			])
		else:
			return transforms.Compose([
				transforms.ToTensor()
			])

def get_dataloader(directory , image_name , label , mode , grayscale = False , thresholding = False , batch_size = 1024 , num_workers = 8):
	transform = get_transform(mode , thresholding)
	dataset = Dataset(directory , image_name , label , transform , grayscale)
	dataloader = DataLoader(dataset , batch_size = batch_size , shuffle = (mode == 'train') , num_workers = num_workers)
	return dataloader

def print_progress(epoch , total_epoch , batch , total_batch , total_time = None):
	if batch < total_batch:
		length = int(50 * batch / total_batch)
		bar = length * '=' + '>' + (49 - length) * ' '
		print('\repoch {}/{} ({}/{}) [{}]'.format(epoch , total_epoch , batch , total_batch , bar) , end = '')
	else:
		data = total_batch
		bar = 50 * '='
		print('\repoch {}/{} ({}/{}) [{}] ({}s)'.format(epoch , total_epoch , batch , total_batch , bar , total_time))