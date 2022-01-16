import argparse
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
			return self.transform(self.data[index]) , torch.LongTensor(self.label[index])
		else:
			return self.transform(self.data[index])

def my_argparse():
	parser = argparse.ArgumentParser()
	parser.add_argument('--train_directory' , type = str , default = 'hw1_data/p2_data/train/')
	parser.add_argument('--validation_directory' , type = str , default = 'hw1_data/p2_data/validation/')
	parser.add_argument('--test_directory' , type = str , default = 'hw1_data/p2_data/validation/')
	parser.add_argument('--output_directory' , type = str , default = 'prediction/')
	parser.add_argument('--checkpoint_directory' , type = str , default = './')
	parser.add_argument('--checkpoint' , type = str , default = 'DeepLabV3_ResNet50.pth')
	parser.add_argument('--optimizer' , type = str , default = 'Adam')
	parser.add_argument('--batch_size' , type = int , default = 4)
	parser.add_argument('--learning_rate' , type = float , default = 0.0002)
	parser.add_argument('--weight_decay' , type = float , default = 0)
	parser.add_argument('--epoch' , type = int , default = 30)
	args = parser.parse_args()
	return args

def get_transform(mode):
	return transforms.Compose([
		transforms.ToTensor() , 
		transforms.Normalize(mean = [0.485 , 0.456 , 0.406] , std = [0.229 , 0.224 , 0.225])
	])

def get_dataloader(data , label , mode , batch_size = 4 , num_workers = 0):
	transform = get_transform(mode)
	dataset = Dataset(data , label , transform)
	dataloader = DataLoader(dataset , batch_size = batch_size , shuffle = (mode == 'train') , num_workers = num_workers)
	return dataloader

def print_progress(epoch , total_epoch , total_data , batch_size , batch , total_batch , total_time = None , loss = None):
	if batch < total_batch:
		data = batch * batch_size
		length = int(50 * data / total_data)
		bar = length * '=' + '>' + (49 - length) * ' '
		print('\repoch {}/{} ({}/{}) [{}]'.format(epoch , total_epoch , data , total_data , bar) , end = '')
	else:
		data = total_data
		bar = 50 * '='
		print('\repoch {}/{} ({}/{}) [{}] ({}s) train loss : {:.8f}'.format(epoch , total_epoch , data , total_data , bar , total_time , loss))