import os
import numpy as np
import torch
from p2_utils import my_argparse , get_dataloader
from p2_data import load_data , save_prediction
from p2_model import ResNet50

def calculate_mIoU(pred , true):
	mIoU = 0
	for i in range(6):
		TP_FP = np.sum(pred == i)
		TP_FN = np.sum(true == i)
		TP = np.sum((pred == i) * (true == i))
		IoU = TP / (TP_FP + TP_FN - TP)
		mIoU += IoU / 6
	return mIoU

def test(test_dataloader , model , device):
	model.to(device)
	model.eval()
	prediction = []
	with torch.no_grad():
		for data in test_dataloader:
			data = data.to(device)
			output = model(data)
			_ , index = torch.max(output , dim = 1)
			prediction.append(index.cpu().detach().numpy())
	prediction = np.concatenate(prediction , axis = 0)
	return prediction

def main(args):
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print('load testing data...')
	test_image_name = load_data(args.test_directory , args.test_annotation , 'test')
	test_dataloader = get_dataloader(args.test_directory , test_image_name , None , 'test')
	print('test model...')
	model = ResNet50()
	model.load_state_dict(torch.load(os.path.join(args.checkpoint_directory , args.checkpoint) , map_location = device))
	prediction = test(test_dataloader , model , device)
	save_prediction(test_image_name , prediction , args.output_file)

if __name__ == '__main__':
	args = my_argparse()
	main(args)