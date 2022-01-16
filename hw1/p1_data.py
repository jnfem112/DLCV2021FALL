import os
import pandas as pd
import numpy as np
from PIL import Image

def load_data(directory , mode):
	if mode != 'test':
		image_names , x , y = [] , [] , []
		for image_name in os.listdir(directory):
			image = Image.open(os.path.join(directory , image_name))
			label = int(image_name.split('_')[0])
			image_names.append(image_name)
			x.append(image)
			y.append(label)
		return image_names , x , y
	else:
		image_names , x = [] , []
		for image_name in os.listdir(directory):
			image = Image.open(os.path.join(directory , image_name))
			image_names.append(image_name)
			x.append(image)
		return image_names , x

def save_prediction(image_name , prediction , output_file):
	df = pd.DataFrame({'image_id' : image_name , 'label' : prediction})
	df.to_csv(output_file , index = False)