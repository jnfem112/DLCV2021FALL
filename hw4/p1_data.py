import os
import pandas as pd
import numpy as np
from PIL import Image

def load_data(directory , annotation , mode):
	df = pd.read_csv(annotation)
	image_names , x , y = df['filename'].values , [] , df['label'].values
	for image_name in image_names:
		fp = Image.open(os.path.join(directory , image_name))
		image = fp.copy()
		fp.close()
		x.append(image)
	mapping = {label : index for index , label in enumerate(y[np.sort(np.unique(y , return_index = True)[1])])}
	y = [mapping[y[i]] for i in range(len(y))]
	return image_names , x , y