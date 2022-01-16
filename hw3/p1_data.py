import pandas as pd

def save_prediction(image_name , prediction , output_file):
	df = pd.DataFrame({'filename' : image_name , 'label' : prediction})
	df.to_csv(output_file , index = False)