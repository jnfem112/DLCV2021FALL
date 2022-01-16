import argparse
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--prediction' , type = str , default = 'prediction.csv')
args = parser.parse_args()

df = pd.read_csv(args.prediction)
count , total = 0 , 0
for image_name , y_pred in zip(df['image_id'].values , df['label'].values):
    y_true = int(image_name.split('_')[0])
    count += (y_pred == y_true)
    total += 1
print('accuracy : {:.5f}'.format(count / total))