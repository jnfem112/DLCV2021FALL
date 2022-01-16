import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from time import time
from p2_utils import my_argparse , get_dataloader , print_progress
from p2_data import load_data
from p2_model import FCN32s_VGG16 , UNet , DeepLabV3_ResNet50

def dice_loss(input , target , epsilon = 1e-6):
	num_classes = input.shape[1]
	A = F.softmax(input , dim = 1).float()
	B = F.one_hot(target , num_classes = num_classes).permute(0 , 3 , 1 , 2).float()
	dice = 0
	for i in range(num_classes):
		a = A[ : , i , : , : ].reshape(-1)
		b = B[ : , i , : , : ].reshape(-1)
		dice += (2 * torch.dot(a , b) + epsilon) / (torch.sum(a) + torch.sum(b) + epsilon)
	return 1 - dice / num_classes

def calculate_mIoU(pred , true):
	mIoU = 0
	for i in range(6):
		TP_FP = np.sum(pred == i)
		TP_FN = np.sum(true == i)
		TP = np.sum((pred == i) * (true == i))
		IoU = TP / (TP_FP + TP_FN - TP)
		mIoU += IoU / 6
	return mIoU

def train(train_x , train_y , validation_x , validation_y , model , device , args):
	train_dataloader = get_dataloader(train_x , train_y , 'train' , args.batch_size)
	model.to(device)
	optimizer = getattr(optim , args.optimizer)(model.parameters() , lr = args.learning_rate , weight_decay = args.weight_decay)
	criterion = nn.CrossEntropyLoss()
	max_mIoU = 0
	for i in range(args.epoch):
		model.train()
		total_loss = 0
		start = time()
		for j , (data , label) in enumerate(train_dataloader):
			data , label = data.to(device) , label.to(device)
			optimizer.zero_grad()
			output = model(data)['out']
			loss = criterion(output , label)
			total_loss += loss.item()
			loss.backward()
			optimizer.step()
			end = time()
			print_progress(i + 1 , args.epoch , len(train_x) , args.batch_size , j + 1 , len(train_dataloader) , int(end - start) , total_loss / len(train_x))

		mIoU = evaluate(validation_x , validation_y , model , device)
		if mIoU > max_mIoU:
			print('save model...')
			torch.save(model.state_dict() , os.path.join(args.checkpoint_directory , args.checkpoint))
			max_mIoU = mIoU

def evaluate(validation_x , validation_y , model , device):
	validation_dataloader = get_dataloader(validation_x , validation_y , 'validation')
	model.to(device)
	model.eval()
	prediction = []
	start = time()
	with torch.no_grad():
		for data , label in validation_dataloader:
			data , label = data.to(device) , label.to(device)
			output = model(data)['out']
			_ , index = torch.max(output , dim = 1)
			prediction.append(index.cpu().detach().numpy())
	prediction = np.concatenate(prediction , axis = 0)
	mIoU = calculate_mIoU(prediction , validation_y)
	end = time()
	print('evaluation ({}s) validation mIoU : {:.5f}'.format(int(end - start) , mIoU))
	return mIoU

def main(args):
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print('load training data...')
	_ , train_x , train_y = load_data(args.train_directory , mode = 'train')
	print('load validation data...')
	_ , validation_x , validation_y = load_data(args.validation_directory , mode = 'validation')
	print('train model...')
	model = DeepLabV3_ResNet50()
	train(train_x , train_y , validation_x , validation_y , model , device , args)

if __name__ == '__main__':
	args = my_argparse()
	main(args)