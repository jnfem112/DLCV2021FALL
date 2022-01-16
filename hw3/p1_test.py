import os
import numpy as np
import torch
from p1_utils import my_argparse , get_dataloader
from p1_data import save_prediction
from p1_model import myViT

def test(image_name , model , device , args):
	test_dataloader = get_dataloader(args.test_directory , image_name , 'test')
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
	image_name = os.listdir(args.test_directory)
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model = myViT()
	model.load_state_dict(torch.load(os.path.join(args.checkpoint_directory , args.checkpoint) , map_location = device))
	prediction = test(image_name , model , device , args)
	save_prediction(image_name , prediction , args.output_file)

if __name__ == '__main__':
	args = my_argparse()
	main(args)