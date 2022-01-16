import argparse
import math
import cv2
from PIL import Image
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from p1_model import myViT

def my_argparse():
	parser = argparse.ArgumentParser()
	parser.add_argument('--input_image' , type = str)
	parser.add_argument('--output_image' , type = str)
	args = parser.parse_args()
	return args

def visulaize_attention_map(model , device , args):
	transform = transforms.Compose([
		transforms.Resize((384 , 384)) , 
		transforms.ToTensor() , 
		transforms.Normalize(mean = [0.5 , 0.5 , 0.5] , std = [0.5 , 0.5 , 0.5])
	])

	fp = Image.open(args.input_image).convert('RGB')
	image = fp.copy()
	fp.close()
	size = image.size
	image = transform(image).unsqueeze(dim = 0)
	image = image.to(device)
	model.to(device)
	model.eval()
	model(image)
	attention_map = torch.mean(model.transformer.blocks[-1].attn.scores.squeeze()[ : , 0 , 1 : ] , dim = 0).cpu().detach().numpy()
	width = int(math.sqrt(attention_map.shape[0]))
	attention_map = cv2.resize(attention_map.reshape((width , width)) , size , interpolation = cv2.INTER_LINEAR)

	fp = Image.open(args.input_image).convert('RGB')
	image = fp.copy()
	fp.close()
	fig = plt.figure(figsize = (24 , 10))
	ax1 = fig.add_subplot(1 , 3 , 1)
	ax1.set_axis_off()
	ax1.set_title('Original Image' , fontsize = 28 , y = 1.02)
	ax1.imshow(image)
	ax2 = fig.add_subplot(1 , 3 , 2)
	ax2.set_axis_off()
	ax2.set_title('Attention Map' , fontsize = 28 , y = 1.02)
	ax2.imshow(attention_map , cmap = 'jet')
	ax3 = fig.add_subplot(1 , 3 , 3)
	ax3.set_axis_off()
	ax3.set_title('Overlay' , fontsize = 28 , y = 1.02)
	ax3.imshow(image.convert('L') , cmap = 'gray')
	ax3.imshow(attention_map , cmap = 'jet' , alpha = 0.4)
	plt.savefig(args.output_image)

def main(args):
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model = myViT()
	model.load_state_dict(torch.load('ViT.pth' , map_location = device))
	visulaize_attention_map(model , device , args)

if __name__ == '__main__':
	args = my_argparse()
	main(args)