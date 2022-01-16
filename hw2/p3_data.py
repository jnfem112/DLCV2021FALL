import os
import pandas as pd

def load_data(directory , annotation):
	if annotation != None:
		df = pd.read_csv(annotation)
		image_names , labels = [] , []
		for image_name , label in zip(df['image_name'].values , df['label'].values):
			image_names.append(image_name)
			labels.append(label)
		return image_names , labels
	else:
		image_names = []
		for image_name in os.listdir(directory):
			image_names.append(image_name)
		return image_names

def save_prediction(image_name , prediction , output_file):
	df = pd.DataFrame({'image_name' : image_name , 'label' : prediction})
	df.to_csv(output_file , index = False)