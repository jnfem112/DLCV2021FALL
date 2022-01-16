import os
import torch
from torch import optim
from torch.optim import Adam
from byol_pytorch import BYOL
from time import time
from p2_utils import my_argparse , get_dataloader , print_progress
from p2_data import load_data
from p2_model import ResNet50

def pretrain(pretrain_dataloader , model , device , args):
	model.to(device)
	learner = BYOL(model , image_size = 128 , hidden_layer = 'avgpool')
	optimizer = Adam(learner.parameters() , lr = args.learning_rate)
	for i in range(args.epoch):
		total_loss = 0
		start = time()
		for j , data in enumerate(pretrain_dataloader):
			data = data.to(device)
			optimizer.zero_grad()
			loss = learner(data)
			total_loss += loss.item()
			loss.backward()
			optimizer.step()
			learner.update_moving_average()
			end = time()
			print_progress(i + 1 , args.epoch , len(os.listdir(args.pretrain_directory)) , args.batch_size , j + 1 , len(pretrain_dataloader) , int(end - start) , total_loss / len(os.listdir(args.pretrain_directory)) , None)
	torch.save(model.state_dict() , os.path.join(args.checkpoint_directory , args.pretrained_checkpoint))

def main(args):
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print('load data...')
	pretrain_image_name = load_data(args.pretrain_directory , None , 'pretrain')
	pretrain_dataloader = get_dataloader(args.pretrain_directory , pretrain_image_name , None , 'pretrain' , args.batch_size)
	print('train model...')
	model = ResNet50()
	pretrain(pretrain_dataloader , model , device , args)

if __name__ == '__main__':
	args = my_argparse()
	main(args)