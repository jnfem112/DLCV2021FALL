import os
import pandas as pd
import numpy as np
from PIL import Image

label2index = {'Alarm_Clock': 0, 'Backpack': 1, 'Batteries': 2, 'Bed': 3, 'Bike': 4, 'Bottle': 5, 'Bucket': 6, 'Calculator': 7, 'Calendar': 8, 'Candles': 9, 'Chair': 10, 'Clipboards': 11, 'Computer': 12, 'Couch': 13, 'Curtains': 14, 'Desk_Lamp': 15, 'Drill': 16, 'Eraser': 17, 'Exit_Sign': 18, 'Fan': 19, 'File_Cabinet': 20, 'Flipflops': 21, 'Flowers': 22, 'Folder': 23, 'Fork': 24, 'Glasses': 25, 'Hammer': 26, 'Helmet': 27, 'Kettle': 28, 'Keyboard': 29, 'Knives': 30, 'Lamp_Shade': 31, 'Laptop': 32, 'Marker': 33, 'Monitor': 34, 'Mop': 35, 'Mouse': 36, 'Mug': 37, 'Notebook': 38, 'Oven': 39, 'Pan': 40, 'Paper_Clip': 41, 'Pen': 42, 'Pencil': 43, 'Postit_Notes': 44, 'Printer': 45, 'Push_Pin': 46, 'Radio': 47, 'Refrigerator': 48, 'Ruler': 49, 'Scissors': 50, 'Screwdriver': 51, 'Shelf': 52, 'Sink': 53, 'Sneakers': 54, 'Soda': 55, 'Speaker': 56, 'Spoon': 57, 'TV': 58, 'Table': 59, 'Telephone': 60, 'ToothBrush': 61, 'Toys': 62, 'Trash_Can': 63, 'Webcam': 64}
index2label = {0: 'Alarm_Clock', 1: 'Backpack', 2: 'Batteries', 3: 'Bed', 4: 'Bike', 5: 'Bottle', 6: 'Bucket', 7: 'Calculator', 8: 'Calendar', 9: 'Candles', 10: 'Chair', 11: 'Clipboards', 12: 'Computer', 13: 'Couch', 14: 'Curtains', 15: 'Desk_Lamp', 16: 'Drill', 17: 'Eraser', 18: 'Exit_Sign', 19: 'Fan', 20: 'File_Cabinet', 21: 'Flipflops', 22: 'Flowers', 23: 'Folder', 24: 'Fork', 25: 'Glasses', 26: 'Hammer', 27: 'Helmet', 28: 'Kettle', 29: 'Keyboard', 30: 'Knives', 31: 'Lamp_Shade', 32: 'Laptop', 33: 'Marker', 34: 'Monitor', 35: 'Mop', 36: 'Mouse', 37: 'Mug', 38: 'Notebook', 39: 'Oven', 40: 'Pan', 41: 'Paper_Clip', 42: 'Pen', 43: 'Pencil', 44: 'Postit_Notes', 45: 'Printer', 46: 'Push_Pin', 47: 'Radio', 48: 'Refrigerator', 49: 'Ruler', 50: 'Scissors', 51: 'Screwdriver', 52: 'Shelf', 53: 'Sink', 54: 'Sneakers', 55: 'Soda', 56: 'Speaker', 57: 'Spoon', 58: 'TV', 59: 'Table', 60: 'Telephone', 61: 'ToothBrush', 62: 'Toys', 63: 'Trash_Can', 64: 'Webcam'}

def load_data(directory , annotation , mode):
	if annotation is not None:
		if mode != 'test':
			df = pd.read_csv(annotation)
			image_names , labels = [] , []
			for image_name , label in zip(df['filename'].values , df['label'].values):
				image_names.append(image_name)
				labels.append(label2index[label])
			return image_names , labels
		else:
			df = pd.read_csv(annotation)
			image_names = []
			for image_name in df['filename'].values:
				image_names.append(image_name)
			return image_names
	else:
		image_names = []
		for image_name in os.listdir(directory):
			image_names.append(image_name)
		return image_names

def save_prediction(image_name , prediction , output_file):
	prediction = [index2label[prediction[i]] for i in range(len(prediction))]
	df = pd.DataFrame({'id' : np.arange(len(prediction)) , 'filename' : image_name , 'label' : prediction})
	df.to_csv(output_file , index = False)