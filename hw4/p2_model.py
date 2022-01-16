import torch.nn as nn
import torchvision

def ResNet50():
	model = torchvision.models.resnet50(pretrained = False)
	model.fc = nn.Linear(2048 , 65 , bias = True)
	return model