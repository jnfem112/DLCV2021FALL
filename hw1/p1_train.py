import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from time import time
from p1_utils import my_argparse , get_dataloader , print_progress
from p1_data import load_data
from p1_model import VGG16 , ResNext101

def train(train_x , train_y , validation_x , validation_y , model , device , args):
	train_dataloader = get_dataloader(train_x , train_y , 'train' , args.batch_size)
	model.to(device)
	optimizer = getattr(optim , args.optimizer)(model.parameters() , lr = args.learning_rate , weight_decay = args.weight_decay)
	scheduler = optim.lr_scheduler.ExponentialLR(optimizer , gamma = 0.9)
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
			print_progress(i + 1 , args.epoch , len(train_x) , args.batch_size , j + 1 , len(train_dataloader) , int(end - start) , total_loss / len(train_x) , count / len(train_x))
		scheduler.step()

		accuracy = evaluate(validation_x , validation_y , model , device)
		if accuracy >= max_accuracy:
			print('save model...')
			torch.save(model.state_dict() , os.path.join(args.checkpoint_directory , args.checkpoint))
			max_accuracy = accuracy

def evaluate(validation_x , validation_y , model , device):
	validation_dataloader = get_dataloader(validation_x , validation_y , 'validation')
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
	print('evaluation ({}s) validation loss : {:.8f} , validation accuracy : {:.5f}'.format(int(end - start) , total_loss / len(validation_x) , count / len(validation_x)))
	return count / len(validation_x)

def main(args):
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print('load training data...')
	_ , train_x , train_y = load_data(args.train_directory , mode = 'train')
	print('load validation data...')
	_ , validation_x , validation_y = load_data(args.validation_directory , mode = 'validation')
	print('train model...')
	model = ResNext101()
	train(train_x , train_y , validation_x , validation_y , model , device , args)

if __name__ == '__main__':
	args = my_argparse()
	main(args)