import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

################################################## FCN32s-VGG16 ##################################################

class FCN32s_VGG16(nn.Module):
	def __init__(self):
		super(FCN32s_VGG16 , self).__init__()

		vgg16 = torchvision.models.vgg16(pretrained = True)
		self.feature = vgg16.features

		self.score = nn.Sequential(
			nn.Conv2d(512 , 4096 , kernel_size = 7) , 
			nn.ReLU(inplace = True) , 
			nn.Dropout(p = 0.5) , 
			nn.Conv2d(4096 , 4096 , kernel_size = 1) , 
			nn.ReLU(inplace = True) , 
			nn.Dropout(p = 0.5) , 
			nn.Conv2d(4096 , 7 , kernel_size = 1)
		)

		self.upsample = nn.ConvTranspose2d(7 , 7 , kernel_size = 128 , stride = 64)

	def forward(self , x):
		size = x.size()
		feature = self.feature(x)
		score = self.score(feature)
		upsample = self.upsample(score)
		return upsample[ : , : , upsample.shape[2] // 2 - size[2] // 2 : upsample.shape[2] // 2 + size[2] // 2 , upsample.shape[3] // 2 - size[3] // 2 : upsample.shape[3] // 2 + size[3] // 2]

################################################## UNet ##################################################

class DoubleConv2d(nn.Module):
	def __init__(self , in_channels , out_channels):
		super(DoubleConv2d , self).__init__()

		self.convolution = nn.Sequential(
			nn.Conv2d(in_channels , out_channels , kernel_size = 3 , padding = 1) ,
			nn.BatchNorm2d(out_channels) ,
			nn.ReLU(inplace = True) ,
			nn.Conv2d(out_channels , out_channels , kernel_size = 3 , padding = 1) ,
			nn.BatchNorm2d(out_channels) ,
			nn.ReLU(inplace = True)
		)

	def forward(self , x):
		x = self.convolution(x)
		return x

class DownSample(nn.Module):
	def __init__(self , in_channels , out_channels):
		super(DownSample , self).__init__()

		self.down_sample = nn.Sequential(
			nn.MaxPool2d(2) ,
			DoubleConv2d(in_channels, out_channels)
		)

	def forward(self , x):
		x = self.down_sample(x)
		return x

class UpSample(nn.Module):
	def __init__(self , in_channels , out_channels):
		super(UpSample , self).__init__()

		self.up_sample = nn.Upsample(scale_factor = 2 , mode = 'bilinear' , align_corners = True)
		self.convolution = DoubleConv2d(in_channels , out_channels)

	def forward(self , x1 , x2):
		x1 = self.up_sample(x1)
		x1 = F.pad(x1 , [math.floor((x2.shape[2] - x1.shape[2]) / 2) , math.ceil((x2.shape[2] - x1.shape[2]) / 2) , math.floor((x2.shape[3] - x1.shape[3]) / 2) , math.ceil((x2.shape[3] - x1.shape[3]) / 2)])
		x = torch.cat([x2 , x1] , dim = 1)
		x = self.convolution(x)
		return x

class UNet(nn.Module):
	def __init__(self):
		super(UNet , self).__init__()

		self.convolution_1 = DoubleConv2d(3 , 64)
		self.down_sample_1 = DownSample(64 , 128)
		self.down_sample_2 = DownSample(128 , 256)
		self.down_sample_3 = DownSample(256 , 512)
		self.down_sample_4 = DownSample(512 , 512)
		self.up_sample_1 = UpSample(1024 , 256)
		self.up_sample_2 = UpSample(512 , 128)
		self.up_sample_3 = UpSample(256 , 64)
		self.up_sample_4 = UpSample(128 , 64)
		self.convolution_2 = nn.Conv2d(64 , 7 , kernel_size = 1)

	def forward(self, x):
		x1 = self.convolution_1(x)
		x2 = self.down_sample_1(x1)
		x3 = self.down_sample_2(x2)
		x4 = self.down_sample_3(x3)
		x5 = self.down_sample_4(x4)
		x = self.up_sample_1(x5 , x4)
		x = self.up_sample_2(x , x3)
		x = self.up_sample_3(x , x2)
		x = self.up_sample_4(x , x1)
		x = self.convolution_2(x)
		return x

################################################## DeepLabV3-ResNet50 ##################################################

def DeepLabV3_ResNet50():
	model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained = False , num_classes = 7)
	return model