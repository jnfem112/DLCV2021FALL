import argparse
import os
import math
import cv2
from PIL import Image
import torch
from torchvision import transforms
from transformers import BertTokenizer
import matplotlib.pyplot as plt

max_length = 128
attention_weight = None

def my_argparse():
	parser = argparse.ArgumentParser()
	parser.add_argument('--input_directory' , type = str)
	parser.add_argument('--output_directory' , type = str)
	args = parser.parse_args()
	return args

def resize(image):
	height , width = image.size
	scale = 299 / max(height , width)
	image = image.resize((int(scale * height) , int(scale * width)))
	return image

def create_caption_and_mask(start_token):
	caption = torch.zeros((1 , max_length) , dtype = torch.long)
	mask = torch.ones((1 , max_length) , dtype = torch.bool)
	caption[ : , 0] = start_token
	mask[ : , 0] = False
	return caption , mask

@torch.no_grad()
def predict(image , model , device):
	transform = transforms.Compose([
		transforms.Lambda(resize) , 
		transforms.ToTensor() , 
		transforms.Normalize(mean = [0.5 , 0.5 , 0.5] , std = [0.5 , 0.5 , 0.5])
	])

	image = transform(image).unsqueeze(0)
	image = image.to(device)

	tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
	start_token = tokenizer.convert_tokens_to_ids(tokenizer._cls_token)
	end_token = tokenizer.convert_tokens_to_ids(tokenizer._sep_token)
	caption , mask = create_caption_and_mask(start_token)
	caption , mask = caption.to(device) , mask.to(device)

	def hook(model , input , output):
		global attention_weight
		attention_weight = output[1].squeeze().detach().cpu().numpy()

	model.to(device)
	model.eval()
	hook_handle = model.transformer.decoder.layers[-1].multihead_attn.register_forward_hook(hook)

	for i in range(max_length - 1):
		output = model(image , caption , mask)
		output = output[ : , i , : ]
		index = torch.argmax(output , axis = -1)
		if index[0] == 102:
			break
		caption[ : , i + 1] = index[0]
		mask[ : , i + 1] = False

	hook_handle.remove()
	
	caption = tokenizer.decode(caption[0].tolist() , skip_special_tokens = True).capitalize()
	return caption

def plot(image , caption , save_path):
	caption = caption[ : -1].lower().split()
	fig = plt.figure(figsize = (16 , 3 * math.ceil((len(caption) + 1) / 5)))
	for i in range(len(caption) + 1):
		result = attention_weight[i].reshape((round(math.sqrt(attention_weight.shape[1] * image.size[1] / image.size[0])) , round(math.sqrt(attention_weight.shape[1] * image.size[0] / image.size[1]))))
		result = cv2.resize(result , (image.size[0] , image.size[1]))
		ax = fig.add_subplot(math.ceil((len(caption) + 1) / 5) , 5 , i + 1)
		ax.set_axis_off()
		ax.imshow(image)
		if i != 0:
			ax.imshow(result , cmap = 'jet' , alpha = 0.5)
		if i == 0:
			ax.set_title('<start>' , fontsize = 16)
		elif i == len(caption):
			ax.set_title('<end>' , fontsize = 16)
		else:
			ax.set_title(caption[i] , fontsize = 16)
	plt.savefig(save_path)

def main(args):
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model = torch.hub.load('saahiluppal/catr' , 'v1' , pretrained = True)
	for image_name in os.listdir(args.input_directory):
		fp = Image.open(os.path.join(args.input_directory , image_name)).convert('RGB')
		image = fp.copy()
		fp.close()
		caption = predict(image , model , device)
		plot(image , caption , os.path.join(args.output_directory , '{}.png'.format(image_name.split('.')[0])))

if __name__ == '__main__':
	args = my_argparse()
	main(args)