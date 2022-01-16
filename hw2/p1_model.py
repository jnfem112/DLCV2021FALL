import os
import torch
from torch.autograd import Variable , grad
import torch.nn as nn
from torch.nn.utils import spectral_norm
import torch.nn.functional as F
from torch.optim import Adam , RMSprop
from torchvision.utils import save_image
from time import time
from p1_utils import Dataset , get_transform , get_dataloader , print_progress
from p1_evaluate import calculate_fid_given_paths , inception_score

def init_weight(layer):
	if layer.__class__.__name__.find('Conv') != -1:
		layer.weight.data.normal_(0 , 0.02)
	elif layer.__class__.__name__.find('BatchNorm') != -1:
		layer.weight.data.normal_(1 , 0.02)
		layer.bias.data.fill_(0)

class Module(nn.Module):
	def evaluate(self , args , number_of_output):
		self.generator.to(self.device)
		self.generator.eval()
		index = 0
		start = time()
		while number_of_output > 0:
			input_size = min(args.batch_size , number_of_output)
			number_of_output -= input_size
			input = Variable(torch.randn(input_size , args.input_dim))
			input = input.to(self.device)
			output = self.generator(input).data
			image = (output + 1) / 2
			for j in range(len(image)):
				save_image(image[j] , os.path.join(args.output_directory , '{:04d}.png'.format(index + 1)))
				index += 1
		FID = calculate_fid_given_paths((args.test_directory , args.output_directory) , 32 , self.device , 2048 , 8)
		IS = inception_score(Dataset(args.output_directory , get_transform()) , cuda = torch.cuda.is_available() , batch_size = 32 , resize = True , splits = 10)
		end = time()
		print('evaluation ({}s) FID : {:.8f} , IS : {:.8f}'.format(int(end - start) , FID , IS))
		return FID , IS

	def inference(self , args , number_of_output , make_grid = False , nrow = 10 , image_name = None):
		self.generator.to(self.device)
		self.generator.eval()
		images , index = [] , 0
		start = time()
		while number_of_output > 0:
			input_size = min(args.batch_size , number_of_output)
			number_of_output -= input_size
			input = Variable(torch.randn(input_size , args.input_dim))
			input = input.to(self.device)
			output = self.generator(input).data
			image = (output + 1) / 2
			if make_grid:
				images.append(image)
			else:
				for j in range(len(image)):
					print('save image : {}'.format(os.path.join(args.output_directory , '{:04d}.png'.format(index + 1))))
					save_image(image[j] , os.path.join(args.output_directory , '{:04d}.png'.format(index + 1)))
					index += 1
		if make_grid:
			images = torch.cat(images)
			save_image(images , image_name , nrow = nrow)
		end = time()
		print('inference ({}s)'.format(int(end - start)))

	def save(self , checkpoint):
		torch.save(self.generator.state_dict() , checkpoint)

	def load(self , checkpoint):
		self.generator.load_state_dict(torch.load(checkpoint , map_location = self.device))

################################################## DCGAN ##################################################

class DCGAN_generator(nn.Module):
	def __init__(self , input_dim):
		super(DCGAN_generator , self).__init__()

		self.linear = nn.Sequential(
			nn.Linear(input_dim , 8192 , bias = False) ,
			nn.BatchNorm1d(8192) ,
			nn.ReLU()
		)

		self.deconvolution = nn.Sequential(
			nn.ConvTranspose2d(512 , 256 , kernel_size = 5 , stride = 2 , padding = 2 , output_padding = 1 , bias = False) ,
			nn.BatchNorm2d(256) ,
			nn.ReLU() ,
			nn.Dropout2d(p = 0.2) , 
			nn.ConvTranspose2d(256 , 128 , kernel_size = 5 , stride = 2 , padding = 2 , output_padding = 1 , bias = False) ,
			nn.BatchNorm2d(128) ,
			nn.ReLU() ,
			nn.Dropout2d(p = 0.2) , 
			nn.ConvTranspose2d(128 , 64 , kernel_size = 5 , stride = 2 , padding = 2 , output_padding = 1 , bias = False) ,
			nn.BatchNorm2d(64) ,
			nn.ReLU() ,
			nn.Dropout2d(p = 0.2) , 
			nn.ConvTranspose2d(64 , 3 , kernel_size = 5 , stride = 2 , padding = 2 , output_padding = 1) ,
			nn.Tanh()
		)

		self.apply(init_weight)

	def forward(self , x):
		x = self.linear(x)
		x = x.view(x.size(dim = 0) , -1 , 4 , 4)
		x = self.deconvolution(x)
		return x

class DCGAN_discriminator(nn.Module):
	def __init__(self):
		super(DCGAN_discriminator , self).__init__()

		self.convolution = nn.Sequential(
			nn.Conv2d(3 , 64 , kernel_size = 5 , stride = 2 , padding = 2) ,
			nn.LeakyReLU(0.2) ,
			nn.Conv2d(64 , 128 , kernel_size = 5 , stride = 2 , padding = 2) ,
			nn.BatchNorm2d(128) ,
			nn.LeakyReLU(0.2) ,
			nn.Conv2d(128 , 256 , kernel_size = 5 , stride = 2 , padding = 2) ,
			nn.BatchNorm2d(256) ,
			nn.LeakyReLU(0.2) ,
			nn.Conv2d(256 , 512 , kernel_size = 5 , stride = 2 , padding = 2) ,
			nn.BatchNorm2d(512) ,
			nn.LeakyReLU(0.2) ,
			nn.Conv2d(512 , 1 , kernel_size = 4) ,
			nn.Sigmoid()
		)

		self.apply(init_weight)

	def forward(self , x):
		x = self.convolution(x)
		x = x.view(-1)
		return x

class DCGAN(Module):
	def __init__(self , input_dim):
		super(DCGAN , self).__init__()
		self.generator = DCGAN_generator(input_dim)
		self.discriminator = DCGAN_discriminator()
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	def train(self , args):
		train_dataloader = get_dataloader(args.train_directory , args.batch_size)
		self.generator = self.generator.to(self.device)
		self.discriminator = self.discriminator.to(self.device)
		optimizer_generator = Adam(self.generator.parameters() , lr = args.learning_rate_generator , betas = (0.5 , 0.999))
		optimizer_discriminator = Adam(self.discriminator.parameters() , lr = args.learning_rate_discriminator , betas = (0.5 , 0.999))
		criterion = nn.BCELoss()
		for i in range(args.epoch):
			self.generator.train()
			self.discriminator.train()
			total_loss_generator = 0
			total_loss_discriminator = 0
			start = time()
			for j , image in enumerate(train_dataloader):
				# Update discriminator.
				input = Variable(torch.randn(image.size(dim = 0) , args.input_dim)).to(self.device)
				real_image = Variable(image).to(self.device)
				fake_image = self.generator(input)
				real_label = torch.ones((image.size(dim = 0))).to(self.device)
				fake_label = torch.zeros((image.size(dim = 0))).to(self.device)
				optimizer_discriminator.zero_grad()
				real_output = self.discriminator(real_image.detach())
				fake_output = self.discriminator(fake_image.detach())
				loss_discriminator = (criterion(real_output , real_label) + criterion(fake_output , fake_label)) / 2
				total_loss_discriminator += loss_discriminator.item()
				loss_discriminator.backward()
				optimizer_discriminator.step()
				# Update generator.
				input = Variable(torch.randn(image.size(dim = 0) , args.input_dim)).to(self.device)
				fake_image = self.generator(input)
				fake_output = self.discriminator(fake_image)
				optimizer_generator.zero_grad()
				loss_generator = criterion(fake_output , real_label)
				total_loss_generator += loss_generator.item()
				loss_generator.backward()
				optimizer_generator.step()
				end = time()
				print_progress(i + 1 , args.epoch , len(os.listdir(args.train_directory)) , args.batch_size , j + 1 , len(train_dataloader) , int(end - start) , total_loss_generator / len(os.listdir(args.train_directory)) , total_loss_discriminator / len(os.listdir(args.train_directory)))

			FID , IS = self.evaluate(args , args.number_of_output)
			if FID < 30 and IS > 2:
				print('save model...')
				self.save('DCGAN_generator_epoch_{}.pth'.format(i + 1))
			self.inference(args , 100 , make_grid = True , nrow = 10 , image_name = f'{args.model}_epoch_{i + 1}.png')