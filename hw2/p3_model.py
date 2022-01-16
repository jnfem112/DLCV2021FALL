import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.optim import Adam
from time import time
from p3_utils import get_dataloader , print_progress
from p3_data import load_data , save_prediction

################################################## DaNN ##################################################

class FeatureExtractor(nn.Module):
	def __init__(self , channel = 3):
		super(FeatureExtractor , self).__init__()

		self.convolution = nn.Sequential(
			nn.Conv2d(channel , 64 , kernel_size = 3 , stride = 1 , padding = 1) ,
			nn.BatchNorm2d(64) ,
			nn.ReLU() ,
			nn.MaxPool2d(2) ,
			nn.Conv2d(64 , 128 , kernel_size = 3 , stride = 1 , padding = 1) ,
			nn.BatchNorm2d(128) ,
			nn.ReLU() ,
			nn.MaxPool2d(2) ,
			nn.Conv2d(128 , 256 , kernel_size = 3 , stride = 1 , padding = 1) ,
			nn.BatchNorm2d(256) ,
			nn.ReLU() ,
			nn.MaxPool2d(2) ,
			nn.Conv2d(256 , 512 , kernel_size = 3 , stride = 1 , padding = 1) ,
			nn.BatchNorm2d(512) ,
			nn.ReLU() ,
			nn.MaxPool2d(2)
		)
		
	def forward(self , x):
		x = self.convolution(x)
		x = x.view(x.size(dim = 0) , -1)
		return x

class LabelPredictor(nn.Module):
	def __init__(self):
		super(LabelPredictor , self).__init__()

		self.linear = nn.Sequential(
			nn.Linear(512 , 512) ,
			nn.BatchNorm1d(512) , 
			nn.ReLU() ,
			nn.Dropout(p = 0.5) , 
			nn.Linear(512 , 512) ,
			nn.BatchNorm1d(512) , 
			nn.ReLU() ,
			nn.Dropout(p = 0.5) , 
			nn.Linear(512 , 10)
		)

	def forward(self , x):
		x = self.linear(x)
		return x

class DomainClassifier(nn.Module):
	def __init__(self):
		super(DomainClassifier , self).__init__()

		self.linear = nn.Sequential(
			nn.Linear(512 , 512) ,
			nn.BatchNorm1d(512) , 
			nn.ReLU() ,
			nn.Dropout(p = 0.5) , 
			nn.Linear(512 , 512) ,
			nn.BatchNorm1d(512) , 
			nn.ReLU() ,
			nn.Dropout(p = 0.5) , 
			nn.Linear(512 , 1) , 
			nn.Sigmoid()
		)

	def forward(self , x):
		x = self.linear(x)
		return x

class DaNN(nn.Module):
	def __init__(self , channel = 3):
		super(DaNN , self).__init__()
		self.feature_extractor = FeatureExtractor(channel)
		self.label_predictor = LabelPredictor()
		self.domain_classifier = DomainClassifier()
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	
	def train(self , args):
		source_image_name , source_label = load_data(args.source_directory , args.source_annotation)
		target_image_name , target_label = load_data(args.target_directory , args.target_annotation)
		source_dataloader = get_dataloader(args.source_directory , source_image_name , source_label , 'train' , args.source == 'usps' , False , args.batch_size)
		target_dataloader = get_dataloader(args.target_directory , target_image_name , target_label , 'train' , args.source == 'usps' , args.source == 'usps' , args.batch_size)
		self.feature_extractor.to(self.device)
		self.label_predictor.to(self.device)
		self.domain_classifier.to(self.device)
		optimizer_feature_extractor = Adam(self.feature_extractor.parameters() , lr = args.learning_rate)
		optimizer_label_predictor = Adam(self.label_predictor.parameters() , lr = args.learning_rate)
		optimizer_domain_classifier = Adam(self.domain_classifier.parameters() , lr = args.learning_rate)
		criterion_label_predictor = nn.CrossEntropyLoss()
		criterion_domain_classifier = nn.BCELoss()
		max_accuracy = 0
		for i in range(args.epoch):
			self.feature_extractor.train()
			self.label_predictor.train()
			self.domain_classifier.train()
			start = time()
			for j , ((source_image , source_label) , (target_image , taget_label)) in enumerate(zip(source_dataloader , target_dataloader)):
				image = torch.cat([source_image , target_image] , dim = 0)
				domain_label = torch.zeros((source_image.shape[0] + target_image.shape[0] , 1))
				domain_label[ : source_image.shape[0]] = 1
				image , domain_label = image.to(self.device) , domain_label.to(self.device)
				source_image , source_label = source_image.to(self.device) , source_label.to(self.device)
				target_image , taget_label = target_image.to(self.device) , taget_label.to(self.device)
				optimizer_feature_extractor.zero_grad()
				optimizer_label_predictor.zero_grad()
				optimizer_domain_classifier.zero_grad()
				# Update domain classifier.
				feature = self.feature_extractor(image)
				output_domain_classifier = self.domain_classifier(feature.detach())
				loss_domain_classifier = criterion_domain_classifier(output_domain_classifier , domain_label)
				loss_domain_classifier.backward()
				optimizer_domain_classifier.step()
				# Update feature extractor and label predictor.
				output_label_predictor = self.label_predictor(feature[ : source_image.shape[0]])
				output_domain_classifier = self.domain_classifier(feature)
				loss_label_predictor = criterion_label_predictor(output_label_predictor , source_label) - args.lambd * criterion_domain_classifier(output_domain_classifier , domain_label)
				loss_label_predictor.backward()
				optimizer_feature_extractor.step()
				optimizer_label_predictor.step()
				end = time()
				print_progress(i + 1 , args.epoch , j + 1 , min(len(source_dataloader) , len(target_dataloader)) , int(end - start))

			accuracy = self.evaluate(args)
			if accuracy >= max_accuracy:
				print('save model...')
				self.save(args)
				max_accuracy = accuracy

	def evaluate(self , args):
		test_image_name , test_label = load_data(args.test_directory , args.test_annotation)
		test_dataloader = get_dataloader(args.test_directory , test_image_name , test_label , 'test' , args.source == 'usps' , args.source == 'usps')
		self.feature_extractor.eval()
		self.label_predictor.eval()
		self.feature_extractor.to(self.device)
		self.label_predictor.to(self.device)
		count = 0
		start = time()
		with torch.no_grad():
			for image , label in test_dataloader:
				image , label = image.to(self.device) , label.to(self.device)
				output = self.label_predictor(self.feature_extractor(image))
				_ , index = torch.max(output , dim = 1)
				count += torch.sum(label == index).item()
		end = time()
		print('evaluation ({}s) accuracy : {:.5f}'.format(int(end - start) , count / len(os.listdir(args.test_directory))))
		return count / len(os.listdir(args.test_directory))

	def test(self , args):
		test_image_name = load_data(args.test_directory , None)
		test_dataloader = get_dataloader(args.test_directory , test_image_name , None , 'test' , args.source == 'usps' , args.source == 'usps')
		self.feature_extractor.eval()
		self.label_predictor.eval()
		self.feature_extractor.to(self.device)
		self.label_predictor.to(self.device)
		prediction = []
		with torch.no_grad():
			for image in test_dataloader:
				image = image.to(self.device)
				output = self.label_predictor(self.feature_extractor(image))
				_ , index = torch.max(output , dim = 1)
				prediction.append(index.cpu().detach().numpy())
		prediction = np.concatenate(prediction , axis = 0)
		save_prediction(test_image_name , prediction , args.output_file)

	def save(self , args):
		torch.save(self.feature_extractor.state_dict() , f'DaNN_feature_extractor_{args.target}.pth')
		torch.save(self.label_predictor.state_dict() , f'DaNN_label_predictor_{args.target}.pth')

	def load(self , args):
		self.feature_extractor.load_state_dict(torch.load(f'DaNN_feature_extractor_{args.target}.pth' , map_location = self.device))
		self.label_predictor.load_state_dict(torch.load(f'DaNN_label_predictor_{args.target}.pth' , map_location = self.device))

################################################## MCD ##################################################

class GradReverse(Function):
	def __init__(self , lambd):
		self.lambd = lambd

	def forward(self , x):
		return x.view_as(x)

	def backward(self , grad_output):
		return -self.lambd * grad_output

def grad_reverse(x , lambd = 1.0):
	return GradReverse(lambd)(x)

class Generator(nn.Module):
	def __init__(self , channel = 3):
		super(Generator , self).__init__()
		self.conv1 = nn.Conv2d(channel , 64 , kernel_size = (5 , 5) , stride = (1 , 1) , padding = (2 , 2))
		self.bn1 = nn.BatchNorm2d(64)
		self.conv2 = nn.Conv2d(64 , 64 , kernel_size = (5 , 5) , stride = (1 , 1) , padding = (2 , 2))
		self.bn2 = nn.BatchNorm2d(64)
		self.conv3 = nn.Conv2d(64 , 128 , kernel_size = (5 , 5) , stride = (1 , 1) , padding = (2 , 2))
		self.bn3 = nn.BatchNorm2d(128)
		self.fc1 = nn.Linear(6272 , 3072)
		self.bn4 = nn.BatchNorm1d(3072)

	def forward(self , x):
		x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))) , stride = (2 , 2) , kernel_size = (3 , 3) , padding = (1 , 1))
		x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))) , stride = (2 , 2) , kernel_size = (3 , 3) , padding = (1 , 1))
		x = F.relu(self.bn3(self.conv3(x)))
		x = x.view(x.size(0) , 6272)
		x = F.relu(self.bn4(self.fc1(x)))
		x = F.dropout(x , training = self.training)
		return x

class Classifier(nn.Module):
	def __init__(self):
		super(Classifier , self).__init__()
		self.fc1 = nn.Linear(3072 , 2048)
		self.bn1 = nn.BatchNorm1d(2048)
		self.fc2 = nn.Linear(2048 , 10)

	def set_lambda(self , lambd):
		self.lambd = lambd

	def forward(self , x , reverse = False):
		if reverse:
			x = grad_reverse(x , self.lambd)
		x = F.relu(self.bn1(self.fc1(x)))
		x = self.fc2(x)
		return x

class MCD(nn.Module):
	def __init__(self , channel = 3):
		super(MCD , self).__init__()
		self.generator = Generator(channel)
		self.classifier_1 = Classifier()
		self.classifier_2 = Classifier()
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	
	def discrepancy(self , output_1 , output_2):
		return torch.mean(torch.abs(F.softmax(output_1 , dim = 1) - F.softmax(output_2 , dim = 1)))
	
	def train(self , args):
		source_image_name , source_label = load_data(args.source_directory , args.source_annotation)
		target_image_name , target_label = load_data(args.target_directory , args.target_annotation)
		source_dataloader = get_dataloader(args.source_directory , source_image_name , source_label , 'train' , args.source == 'usps' , False , args.batch_size)
		target_dataloader = get_dataloader(args.target_directory , target_image_name , target_label , 'train' , args.source == 'usps' , args.source == 'usps' , args.batch_size)
		self.generator.to(self.device)
		self.classifier_1.to(self.device)
		self.classifier_2.to(self.device)
		optimizer_generator = Adam(self.generator.parameters() , lr = args.learning_rate)
		optimizer_classifier_1 = Adam(self.classifier_1.parameters() , lr = args.learning_rate)
		optimizer_classifier_2 = Adam(self.classifier_2.parameters() , lr = args.learning_rate)
		max_accuracy = 0
		for i in range(args.epoch):
			self.generator.train()
			self.classifier_1.train()
			self.classifier_2.train()
			start = time()
			for j , ((source_image , source_label) , (target_image , target_label)) in enumerate(zip(source_dataloader , target_dataloader)):
				source_image , source_label = source_image.to(self.device) , source_label.to(self.device)
				target_image , target_label = target_image.to(self.device) , target_label.to(self.device)
				# Step 1
				optimizer_generator.zero_grad()
				optimizer_classifier_1.zero_grad()
				optimizer_classifier_2.zero_grad()
				feature = self.generator(source_image)
				output_1 = self.classifier_1(feature)
				output_2 = self.classifier_2(feature)
				loss = F.cross_entropy(output_1 , source_label) + F.cross_entropy(output_2 , source_label)
				loss.backward()
				optimizer_generator.step()
				optimizer_classifier_1.step()
				optimizer_classifier_2.step()
				# Step 2
				optimizer_generator.zero_grad()
				optimizer_classifier_1.zero_grad()
				optimizer_classifier_2.zero_grad()
				feature = self.generator(source_image)
				output_1 = self.classifier_1(feature)
				output_2 = self.classifier_2(feature)
				loss_1 = F.cross_entropy(output_1 , source_label) + F.cross_entropy(output_2 , source_label)
				feature = self.generator(target_image)
				output_1 = self.classifier_1(feature)
				output_2 = self.classifier_2(feature)
				loss_2 = self.discrepancy(output_1 , output_2)
				loss = loss_1 - loss_2
				loss.backward()
				optimizer_classifier_1.step()
				optimizer_classifier_2.step()
				# Step 3
				for k in range(4):
					feature = self.generator(target_image)
					output_1 = self.classifier_1(feature)
					output_2 = self.classifier_2(feature)
					loss = self.discrepancy(output_1 , output_2)
					loss.backward()
					optimizer_generator.step()
				end = time()
				print_progress(i + 1 , args.epoch , j + 1 , min(len(source_dataloader) , len(target_dataloader)) , int(end - start))

			accuracy = self.evaluate(args)
			if accuracy >= max_accuracy:
				print('save model...')
				self.save(args)
				max_accuracy = accuracy

	def evaluate(self , args):
		test_image_name , test_label = load_data(args.test_directory , args.test_annotation)
		test_dataloader = get_dataloader(args.test_directory , test_image_name , test_label , 'test' , args.source == 'usps' , args.source == 'usps')
		self.generator.to(self.device)
		self.classifier_1.to(self.device)
		self.classifier_2.to(self.device)
		self.generator.eval()
		self.classifier_1.eval()
		self.classifier_2.eval()
		count = 0
		start = time()
		with torch.no_grad():
			for image , label in test_dataloader:
				image , label = image.to(self.device) , label.to(self.device)
				feature = self.generator(image)
				output_1 = self.classifier_1(feature)
				output_2 = self.classifier_2(feature)
				output = (output_1 + output_2) / 2
				_ , index = torch.max(output , dim = 1)
				count += torch.sum(label == index).item()
		end = time()
		print('evaluation ({}s) accuracy : {:.5f}'.format(int(end - start) , count / len(os.listdir(args.test_directory))))
		return count / len(os.listdir(args.test_directory))

	def test(self , args):
		test_image_name = load_data(args.test_directory , None)
		test_dataloader = get_dataloader(args.test_directory , test_image_name , None , 'test' , args.source == 'usps' , args.source == 'usps')
		self.generator.to(self.device)
		self.classifier_1.to(self.device)
		self.classifier_2.to(self.device)
		self.generator.eval()
		self.classifier_1.eval()
		self.classifier_2.eval()
		prediction = []
		with torch.no_grad():
			for image in test_dataloader:
				image = image.to(self.device)
				feature = self.generator(image)
				output_1 = self.classifier_1(feature)
				output_2 = self.classifier_2(feature)
				output = (output_1 + output_2) / 2
				_ , index = torch.max(output , dim = 1)
				prediction.append(index.cpu().detach().numpy())
		prediction = np.concatenate(prediction , axis = 0)
		save_prediction(test_image_name , prediction , args.output_file)

	def save(self , args):
		torch.save(self.generator.state_dict() , f'MCD_generator_{args.target}.pth')
		torch.save(self.classifier_1.state_dict() , f'MCD_classifier_{args.target}_1.pth')
		torch.save(self.classifier_2.state_dict() , f'MCD_classifier_{args.target}_2.pth')

	def load(self , args):
		self.generator.load_state_dict(torch.load(f'MCD_generator_{args.target}.pth' , map_location = self.device))
		self.classifier_1.load_state_dict(torch.load(f'MCD_classifier_{args.target}_1.pth' , map_location = self.device))
		self.classifier_2.load_state_dict(torch.load(f'MCD_classifier_{args.target}_2.pth' , map_location = self.device))