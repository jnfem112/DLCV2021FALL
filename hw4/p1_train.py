import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from time import time
from p1_utils import my_argparse , get_dataloader , print_progress
from p1_data import load_data
from p1_model import PrototypicalNetwork

def train(train_dataloader , validation_dataloader , model , device , args):
	model.to(device)
	optimizer = Adam(model.parameters() , lr = args.learning_rate , weight_decay = args.weight_decay)
	scheduler = StepLR(optimizer , gamma = 0.5 , step_size = 20)
	max_accuracy = 0
	for i in range(args.epoch):
		model.train()
		total_loss , total_accuracy = 0 , 0
		start = time()
		for j , (data , label) in enumerate(train_dataloader):
			data , label = data.to(device) , label.to(device)
			optimizer.zero_grad()
			feature = model(data)
			loss , accuracy = model.prototypical_loss(feature , target = label , n_support = args.train_shot)
			total_loss += loss.item()
			total_accuracy += accuracy.item()
			loss.backward()
			optimizer.step()
			end = time()
			print_progress(i + 1 , args.epoch , j + 1 , args.train_episode , int(end - start) , total_loss / len(train_dataloader) , total_accuracy / len(train_dataloader))
		scheduler.step()

		accuracy = evaluate(validation_dataloader , model , device , args)
		if accuracy >= max_accuracy:
			print('save model...')
			torch.save(model.state_dict() , os.path.join(args.checkpoint_directory , args.checkpoint))
			max_accuracy = accuracy

def evaluate(validation_dataloader , model , device , args):
	model.to(device)
	model.eval()
	total_loss , total_accuracy = 0 , 0
	start = time()
	with torch.no_grad():
		for data , label in validation_dataloader:
			data , label = data.to(device) , label.to(device)
			feature = model(data)
			loss , accuracy = model.prototypical_loss(feature , target = label , n_support = args.eval_shot)
			total_loss += loss.item()
			total_accuracy += accuracy.item()
	end = time()
	print('evaluation ({}s) validation loss : {:.8f} , validation accuracy : {:.5f}'.format(int(end - start) , total_loss / len(validation_dataloader) , total_accuracy / len(validation_dataloader)))
	return total_accuracy / len(validation_dataloader)

def main(args):
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print('load training data...')
	_ , train_x , train_y = load_data(args.train_directory , args.train_annotation , 'train')
	train_dataloader = get_dataloader(train_x , train_y , args , 'train')
	print('load validation data...')
	_ , validation_x , validation_y = load_data(args.validation_directory , args.validation_annotation , 'validation')
	validation_dataloader = get_dataloader(validation_x , validation_y , args , 'validation')
	print('train model...')
	model = PrototypicalNetwork()
	train(train_dataloader , validation_dataloader , model , device , args)

if __name__ == '__main__':
	args = my_argparse()
	main(args)