#!/bin/bash

:<<!
	python3 p1_train.py
!

wget https://www.dropbox.com/s/cwj0sya07lrtdje/ViT.pth?dl=0 -O ViT.pth
python3 p1_test.py --test_directory=$1 --output_file=$2