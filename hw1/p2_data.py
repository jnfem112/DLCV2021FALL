import os
import numpy as np
from PIL import Image

mapping = {
	0 : [0 , 255 , 255] ,
	1 : [255 , 255 , 0] ,
	2 : [255 , 0 , 255] ,
	3 : [0 , 255 , 0] ,
	4 : [0 , 0 , 255] ,
	5 : [255 , 255 , 255] ,
	6 : [0 , 0 , 0]
}

def load_data(directory , mode):
	if mode != 'test':
		image_names , x , y = [] , [] , []
		for index in range(len(os.listdir(directory)) // 2):
			image_name = '{:04d}_sat.jpg'.format(index)
			mask_name = '{:04d}_mask.png'.format(index)
			image = np.array(Image.open(os.path.join(directory , image_name)))
			mask = np.array(Image.open(os.path.join(directory , mask_name)))
			label = np.zeros((image.shape[0] , image.shape[1]))
			for key , value in mapping.items():
				label[(mask[ : , : , 0] == value[0]) * (mask[ : , : , 1] == value[1]) * (mask[ : , : , 2] == value[2])] = key
			image_names.append(image_name)
			x.append(image)
			y.append(label)
		return image_names , np.array(x) , np.array(y)
	else:
		image_names , x = [] , []
		for image_name in os.listdir(directory):
			image = np.array(Image.open(os.path.join(directory , image_name)))
			image_names.append(image_name)
			x.append(image)
		return image_names , np.array(x)

def save_prediction(image_names , predictions , output_directory):
	for image_name , prediction in zip(image_names , predictions):
		mask = np.zeros((prediction.shape[0] , prediction.shape[1] , 3))
		for i in range(7):
			mask[prediction == i] = mapping[i]
		mask = Image.fromarray(mask.astype(np.uint8))
		if not image_name.endswith('png'):
			image_name = '{}.png'.format(image_name.split('.')[0])
		mask.save(os.path.join(output_directory , image_name))