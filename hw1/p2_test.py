import os
import numpy as np
import torch
from p2_utils import my_argparse , get_dataloader
from p2_data import load_data , save_prediction
from p2_model import FCN32s_VGG16 , UNet , DeepLabV3_ResNet50

def calculate_mIoU(pred , true):
	mIoU = 0
	for i in range(6):
		TP_FP = np.sum(pred == i)
		TP_FN = np.sum(true == i)
		TP = np.sum((pred == i) * (true == i))
		IoU = TP / (TP_FP + TP_FN - TP)
		mIoU += IoU / 6
	return mIoU

def test(test_x , model , device):
	test_dataloader = get_dataloader(test_x , None , 'test')
	model.to(device)
	model.eval()
	test_y = []
	with torch.no_grad():
		for data in test_dataloader:
			data = data.to(device)
			output = model(data)['out']
			_ , index = torch.max(output , dim = 1)
			test_y.append(index.cpu().detach().numpy())
	test_y = np.concatenate(test_y , axis = 0)
	return test_y

def main(args):
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print('load testing data...')
	image_name , test_x = load_data(args.test_directory , mode = 'test')
	print('test model...')
	model = DeepLabV3_ResNet50()
	model.load_state_dict(torch.load(os.path.join(args.checkpoint_directory , args.checkpoint) , map_location = device))
	test_y = test(test_x , model , device)
	save_prediction(image_name , test_y , args.output_directory)

if __name__ == '__main__':
	args = my_argparse()
	main(args)