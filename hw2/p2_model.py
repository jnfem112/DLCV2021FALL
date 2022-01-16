import os
import torch
from torch.autograd import Variable , grad
import torch.nn as nn
from torch.nn.utils import spectral_norm
import torch.nn.functional as F
from torch.optim import Adam
from torchvision.utils import save_image
from time import time
from p2_utils import Dataset , get_transform , get_dataloader , print_progress

def init_weight(layer):
	if layer.__class__.__name__.find('Conv') != -1:
		layer.weight.data.normal_(0 , 0.02)
	elif layer.__class__.__name__.find('BatchNorm') != -1:
		layer.weight.data.normal_(1 , 0.02)
		layer.bias.data.fill_(0)

class Classifier(nn.Module):
	def __init__(self):
		super().__init__()
		self.conv1 = nn.Conv2d(3, 6, 5)
		self.pool = nn.MaxPool2d(2, 2)
		self.conv2 = nn.Conv2d(6, 16, 5)
		self.fc1 = nn.Linear(16 * 4 * 4, 128)
		self.fc2 = nn.Linear(128, 64)
		self.fc3 = nn.Linear(64, 10)

	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		x = torch.flatten(x, 1) # flatten all dimensions except batch
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x

def load_checkpoint(checkpoint_path, model):
	state = torch.load(checkpoint_path, map_location = "cuda")
	model.load_state_dict(state['state_dict'])
	# print('model loaded from %s' % checkpoint_path)

class Module(nn.Module):
	def evaluate(self , args , number_of_total_output):
		self.generator.to(self.device)
		self.generator.eval()
		start = time()
		for i in range(args.number_of_class):
			index = 0
			number_of_output = number_of_total_output // args.number_of_class
			while number_of_output > 0:
				input_size = min(args.batch_size , number_of_output)
				number_of_output -= input_size
				label = i * torch.ones((input_size,) , dtype = torch.long)
				input = Variable(torch.randn(input_size , args.input_dim))
				input = torch.cat([input , F.one_hot(label , num_classes = args.number_of_class)] , dim = 1)
				input = input.to(self.device)
				output = self.generator(input).data
				image = (output + 1) / 2
				for j in range(len(image)):
					save_image(image[j] , os.path.join(args.output_directory , '{}_{:03d}.png'.format(i , index + 1)))
					index += 1

		test_dataloader = get_dataloader(args.output_directory)
		classifier = Classifier()
		load_checkpoint('Classifier.pth' , classifier)
		classifier.to(self.device)
		classifier.eval()
		count = 0
		with torch.no_grad():
			for data , label in test_dataloader:
				data , label = data.to(self.device) , label.to(self.device)
				output = classifier(data)
				_ , index = torch.max(output , dim = 1)
				count += torch.sum(label == index).item()
		end = time()
		print('evaluation ({}s) accuracy : {:.5f}'.format(int(end - start) , count / args.number_of_output))
		return count / args.number_of_output

	def inference(self , args , number_of_total_output , make_grid = False , nrow = 10 , image_name = None):
		self.generator.to(self.device)
		self.generator.eval()
		images = []
		start = time()
		for i in range(args.number_of_class):
			index = 0
			number_of_output = number_of_total_output // args.number_of_class
			while number_of_output > 0:
				input_size = min(args.batch_size , number_of_output)
				number_of_output -= input_size
				label = i * torch.ones((input_size,) , dtype = torch.long)
				input = Variable(torch.randn(input_size , args.input_dim))
				input = torch.cat([input , F.one_hot(label , num_classes = args.number_of_class)] , dim = 1)
				input = input.to(self.device)
				output = self.generator(input).data
				image = (output + 1) / 2
				if make_grid:
					images.append(image)
				else:
					for j in range(len(image)):
						print('save image : {}'.format(os.path.join(args.output_directory , '{}_{:03d}.png'.format(i , index + 1))))
						save_image(image[j] , os.path.join(args.output_directory , '{}_{:03d}.png'.format(i , index + 1)))
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

################################################## ACGAN ##################################################

class ACGAN_generator(nn.Module):
	def __init__(self , input_dim , number_of_class):
		super(ACGAN_generator , self).__init__()

		self.linear = nn.Sequential(
			nn.Linear(input_dim + number_of_class , 8192 , bias = False) ,
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
			nn.ConvTranspose2d(128 , 3 , kernel_size = 4 , stride = 2 , padding = 3 , output_padding = 0 , bias = False) ,
			nn.Tanh()
		)

		self.apply(init_weight)

	def forward(self , x):
		x = self.linear(x)
		x = x.view(x.size(dim = 0) , -1 , 4 , 4)
		x = self.deconvolution(x)
		return x

class ACGAN_discriminator(nn.Module):
	def __init__(self , number_of_class):
		super(ACGAN_discriminator , self).__init__()

		self.convolution = nn.Sequential(
			nn.Conv2d(3 , 64 , kernel_size = 5 , stride = 2 , padding = 2) ,
			nn.BatchNorm2d(64) ,
			nn.LeakyReLU(0.2) ,
			nn.Dropout2d(p = 0.2) , 
			nn.Conv2d(64 , 128 , kernel_size = 5 , stride = 2 , padding = 2) ,
			nn.BatchNorm2d(128) ,
			nn.LeakyReLU(0.2) ,
			nn.Dropout2d(p = 0.2) , 
			nn.Conv2d(128 , 256 , kernel_size = 5 , stride = 2 , padding = 2) ,
			nn.BatchNorm2d(256) ,
			nn.LeakyReLU(0.2) ,
			nn.Dropout2d(p = 0.2) , 
			nn.Conv2d(256 , 512 , kernel_size = 5 , stride = 2 , padding = 2) ,
			nn.BatchNorm2d(512) ,
			nn.LeakyReLU(0.2) ,
			nn.Dropout2d(p = 0.2) , 
			nn.Conv2d(512 , 1024 , kernel_size = 5 , stride = 2 , padding = 2)
		)

		self.discriminator = nn.Sequential(
			nn.Linear(1024 , 1 , bias = True) , 
			nn.Sigmoid()
		)

		self.classifier = nn.Linear(1024 , number_of_class , bias = True)

		self.apply(init_weight)

	def forward(self , x):
		x = self.convolution(x)
		x = x.view(x.size(dim = 0) , -1)
		return self.discriminator(x).squeeze() , self.classifier(x)

class ACGAN(Module):
	def __init__(self , input_dim , number_of_class):
		super(ACGAN , self).__init__()
		self.generator = ACGAN_generator(input_dim , number_of_class)
		self.discriminator = ACGAN_discriminator(number_of_class)
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	def train(self , args):
		train_dataloader = get_dataloader(args.train_directory , args.train_annotation , args.batch_size)
		self.generator = self.generator.to(self.device)
		self.discriminator = self.discriminator.to(self.device)
		optimizer_generator = Adam(self.generator.parameters() , lr = args.learning_rate_generator , betas = (0.5 , 0.999))
		optimizer_discriminator = Adam(self.discriminator.parameters() , lr = args.learning_rate_discriminator , betas = (0.5 , 0.999))
		criterion_discriminator = nn.BCELoss()
		criterion_classifier = nn.CrossEntropyLoss()
		max_accuracy = 0
		for i in range(args.epoch):
			self.generator.train()
			self.discriminator.train()
			total_loss_generator = 0
			total_loss_discriminator = 0
			start = time()
			for j , (image , label) in enumerate(train_dataloader):
				# Update discriminator with real image.
				real_image = Variable(image).to(self.device)
				real_label_discriminator = torch.ones((image.size(dim = 0))).to(self.device)
				real_label_classifier = label.to(self.device)
				optimizer_discriminator.zero_grad()
				real_output_discriminator , real_output_classifier = self.discriminator(real_image.detach())
				loss_discriminator = criterion_discriminator(real_output_discriminator , real_label_discriminator) + criterion_classifier(real_output_classifier , real_label_classifier)
				total_loss_discriminator += loss_discriminator.item()
				loss_discriminator.backward()
				optimizer_discriminator.step()
				# Update discriminator with fake image.
				fake_label_discriminator = torch.zeros((image.size(dim = 0))).to(self.device)
				fake_label_classifier = torch.randint(10 , (image.size(dim = 0),) , dtype = torch.long).to(self.device)
				input = Variable(torch.randn(image.size(dim = 0) , args.input_dim)).to(self.device)
				input = torch.cat([input , F.one_hot(fake_label_classifier , num_classes = args.number_of_class)] , dim = 1)
				fake_image = self.generator(input)
				optimizer_discriminator.zero_grad()
				fake_output_discriminator , fake_output_classifier = self.discriminator(fake_image.detach())
				loss_discriminator = criterion_discriminator(fake_output_discriminator , fake_label_discriminator) + criterion_classifier(fake_output_classifier , fake_label_classifier)
				total_loss_discriminator += loss_discriminator.item()
				loss_discriminator.backward()
				optimizer_discriminator.step()
				# Update generator.
				fake_label_classifier = torch.randint(10 , (image.size(dim = 0),) , dtype = torch.long).to(self.device)
				input = Variable(torch.randn(image.size(dim = 0) , args.input_dim)).to(self.device)
				input = torch.cat([input , F.one_hot(fake_label_classifier , num_classes = args.number_of_class)] , dim = 1)
				fake_image = self.generator(input)
				fake_output_discriminator , fake_output_classifier = self.discriminator(fake_image)
				optimizer_generator.zero_grad()
				loss_generator = criterion_discriminator(fake_output_discriminator , real_label_discriminator) + criterion_classifier(fake_output_classifier , fake_label_classifier)
				total_loss_generator += loss_generator.item()
				loss_generator.backward()
				optimizer_generator.step()
				end = time()
				print_progress(i + 1 , args.epoch , len(os.listdir(args.train_directory)) , args.batch_size , j + 1 , len(train_dataloader) , int(end - start) , total_loss_generator / len(os.listdir(args.train_directory)) , total_loss_discriminator / len(os.listdir(args.train_directory)))

			accuracy = self.evaluate(args , args.number_of_output)
			if accuracy >= max_accuracy:
				print('save model...')
				self.save(args.checkpoint)
				max_accuracy = accuracy
			self.inference(args , 100 , make_grid = True , nrow = 10 , image_name = f'{args.model}_epoch_{i + 1}.png')