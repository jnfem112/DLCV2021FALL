#!/bin/bash

:<<!
python3 p2_train.py
mkdir prediction/
python3 p2_test.py --extention=jpg
python3 mean_iou_evaluate.py --labels=hw1_data/p2_data/validation/ --pred=prediction/
rm -rf prediction/
!

wget -O DeepLabV3_ResNet50.pth https://www.dropbox.com/s/enrhak4dp4aox17/DeepLabV3_ResNet50.pth?dl=0
python3 p2_test.py --test_directory=$1 --output_directory=$2