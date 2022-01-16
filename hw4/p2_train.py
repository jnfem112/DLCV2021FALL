import os
import numpy as np
import torch
from torch._C import device
import torch.nn as nn
from torch.optim import SGD , Adam
from torch.optim.lr_scheduler import ExponentialLR
import torchvision
from time import time
from p2_utils import my_argparse , get_dataloader , print_progress
from p2_data import load_data
from p2_model import ResNet50

def train(train_dataloader , validation_dataloader , model , device , args):
	model.to(device)
	optimizer = Adam(model.parameters() , lr = args.learning_rate , weight_decay = args.weight_decay)
	scheduler = ExponentialLR(optimizer , gamma = 0.9)
	criterion = nn.CrossEntropyLoss()
	max_accuracy = 0
	for i in range(args.epoch):
		model.train()
		count = 0
		total_loss = 0
		start = time()
		for j , (data , label) in enumerate(train_dataloader):
			data , label = data.to(device) , label.to(device)
			optimizer.zero_grad()
			output = model(data)
			_ , index = torch.max(output , dim = 1)
			count += torch.sum(label == index).item()
			loss = criterion(output , label)
			total_loss += loss.item()
			loss.backward()
			optimizer.step()
			end = time()
			print_progress(i + 1 , args.epoch , len(os.listdir(args.train_directory)) , args.batch_size , j + 1 , len(train_dataloader) , int(end - start) , total_loss / len(os.listdir(args.train_directory)) , count / len(os.listdir(args.train_directory)))
		scheduler.step()

		accuracy = evaluate(validation_dataloader , model , device)
		if accuracy >= max_accuracy:
			print('save model...')
			torch.save(model.state_dict() , os.path.join(args.checkpoint_directory , args.checkpoint))
			max_accuracy = accuracy

def evaluate(validation_dataloader , model , device):
	model.to(device)
	model.eval()
	criterion = nn.CrossEntropyLoss()
	count = 0
	total_loss = 0
	start = time()
	with torch.no_grad():
		for data , label in validation_dataloader:
			data , label = data.to(device) , label.to(device)
			output = model(data)
			_ , index = torch.max(output , dim = 1)
			count += torch.sum(label == index).item()
			loss = criterion(output , label)
			total_loss += loss.item()
	end = time()
	print('evaluation ({}s) validation loss : {:.8f} , validation accuracy : {:.5f}'.format(int(end - start) , total_loss / len(os.listdir(args.validation_directory)) , count / len(os.listdir(args.validation_directory))))
	return count / len(os.listdir(args.validation_directory))

def main(args):
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print('load training data...')
	train_image_name , train_label = load_data(args.train_directory , args.train_annotation , 'train')
	train_dataloader = get_dataloader(args.train_directory , train_image_name , train_label , 'train' , args.batch_size)
	print('load validation data...')
	validation_image_name , validation_label = load_data(args.validation_directory , args.validation_annotation , 'validation')
	validation_dataloader = get_dataloader(args.validation_directory , validation_image_name , validation_label , 'validation')
	print('train model...')
	model = ResNet50()
	model.load_state_dict(torch.load(os.path.join(args.checkpoint_directory , args.pretrained_checkpoint) , map_location = device))
	train(train_dataloader , validation_dataloader , model , device , args)

if __name__ == '__main__':
	args = my_argparse()
	main(args)