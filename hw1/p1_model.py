import torch.nn as nn
import torchvision

def VGG16():
	model = torchvision.models.vgg16_bn(pretrained = True)
	model.classifier[-1] = nn.Linear(4096 , 50 , bias = True)
	return model

def ResNext101():
	model = torchvision.models.resnext101_32x8d(pretrained = True)
	model.fc = nn.Linear(2048 , 50 , bias = True)
	return model