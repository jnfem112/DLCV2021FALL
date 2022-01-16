import os
import numpy as np
import torch
from p1_utils import my_argparse , get_dataloader
from p1_data import load_data , save_prediction
from p1_model import VGG16 , ResNext101

def test(test_x , model , device):
	test_dataloader = get_dataloader(test_x , None , 'test')
	model.to(device)
	model.eval()
	test_y = []
	with torch.no_grad():
		for data in test_dataloader:
			data = data.to(device)
			output = model(data)
			_ , index = torch.max(output , dim = 1)
			test_y.append(index.cpu().detach().numpy())
	test_y = np.concatenate(test_y , axis = 0)
	return test_y

def main(args):
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print('load testing data...')
	image_name , test_x = load_data(args.test_directory , mode = 'test')
	print('test model...')
	count = np.zeros((len(test_x) , 50))
	for file_name in os.listdir(args.checkpoint_directory):
		if not file_name.endswith('.pth'):
			continue
		model = ResNext101()
		model.load_state_dict(torch.load(os.path.join(args.checkpoint_directory , file_name) , map_location = device))
		test_y = test(test_x , model , device)
		for i in range(len(test_y)):
			count[i][test_y[i]] += 1
	test_y = count.argmax(axis = 1)
	save_prediction(image_name , test_y , args.output_file)

if __name__ == '__main__':
	args = my_argparse()
	main(args)