import torch.nn as nn
from pytorch_pretrained_vit import ViT

def myViT():
	model = ViT('B_16_imagenet1k' , pretrained = True)
	model.fc = nn.Linear(768 , 37 , bias = True)
	return model