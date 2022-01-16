import argparse
import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset , DataLoader

class Dataset(Dataset):
	def __init__(self , directory , image_name , label , transform):
		self.directory = directory
		self.image_name = image_name
		self.label = label
		self.transform = transform

	def __len__(self):
		return len(self.image_name)

	def __getitem__(self , index):
		fp = Image.open(os.path.join(self.directory , self.image_name[index]))
		image = fp.copy()
		fp.close()
		if self.label is not None:
			return self.transform(image) , self.label[index]
		else:
			return self.transform(image)

def my_argparse():
	parser = argparse.ArgumentParser()
	parser.add_argument('--pretrain_directory' , type = str , default = 'hw4_data/mini/train/')
	parser.add_argument('--train_directory' , type = str , default = 'hw4_data/office/train/')
	parser.add_argument('--train_annotation' , type = str , default = 'hw4_data/office/train.csv')
	parser.add_argument('--validation_directory' , type = str , default = 'hw4_data/office/val/')
	parser.add_argument('--validation_annotation' , type = str , default = 'hw4_data/office/val.csv')
	parser.add_argument('--test_directory' , type = str , default = 'hw4_data/office/val/')
	parser.add_argument('--test_annotation' , type = str , default = 'hw4_data/office/val.csv')
	parser.add_argument('--output_file' , type = str , default = 'prediction.csv')
	parser.add_argument('--checkpoint_directory' , type = str , default = './')
	parser.add_argument('--pretrained_checkpoint' , type = str , default = 'pretrained_ResNet50.pth')
	parser.add_argument('--checkpoint' , type = str , default = 'ResNet50.pth')
	parser.add_argument('--batch_size' , type = int , default = 16)
	parser.add_argument('--learning_rate' , type = float , default = 0.0005)
	parser.add_argument('--weight_decay' , type = float , default = 0)
	parser.add_argument('--epoch' , type = int , default = 50)
	args = parser.parse_args()
	return args

def get_transform(mode):
	if mode == 'train':
		return transforms.Compose([
			transforms.Resize((128 , 128)) , 
			transforms.RandomHorizontalFlip() ,
			transforms.RandomAffine(10 , translate = (0.1 , 0.1) , scale = (0.9 , 1.1)) ,
			transforms.ColorJitter(brightness = 0.15 , contrast = 0.15 , saturation = 0.15 , hue = 0.15) , 
			transforms.ToTensor() , 
			transforms.Normalize(mean = [0.485 , 0.456 , 0.406] , std = [0.229 , 0.224 , 0.225])
		])
	else:
		return transforms.Compose([
			transforms.Resize((128 , 128)) , 
			transforms.ToTensor() , 
			transforms.Normalize(mean = [0.485 , 0.456 , 0.406] , std = [0.229 , 0.224 , 0.225])
		])

def get_dataloader(directory , data , label , mode , batch_size = 16 , num_workers = 8):
	transform = get_transform(mode)
	dataset = Dataset(directory , data , label , transform)
	dataloader = DataLoader(dataset , batch_size = batch_size , shuffle = (mode == 'train') , num_workers = num_workers)
	return dataloader

def print_progress(epoch , total_epoch , total_data , batch_size , batch , total_batch , total_time = None , loss = None , accuracy = None):
	if batch < total_batch:
		data = batch * batch_size
		length = int(50 * data / total_data)
		bar = length * '=' + '>' + (49 - length) * ' '
		print('\repoch {}/{} ({}/{}) [{}]'.format(epoch , total_epoch , data , total_data , bar) , end = '')
	else:
		data = total_data
		bar = 50 * '='
		if accuracy is None:
			print('\repoch {}/{} ({}/{}) [{}] ({}s) train loss : {:.8f}'.format(epoch , total_epoch , data , total_data , bar , total_time , loss))
		else:
			print('\repoch {}/{} ({}/{}) [{}] ({}s) train loss : {:.8f} , train accuracy : {:.5f}'.format(epoch , total_epoch , data , total_data , bar , total_time , loss , accuracy))