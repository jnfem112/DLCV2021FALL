#!/bin/bash

:<<!
for ((i = 0 ; i < 5 ; i++))
do
	python3 p1_train.py --checkpoint=ResNext101_${i}.pth
done
python3 p1_test.py
python3 p1_evaluate.py
!

wget -O ResNext101_0.pth https://www.dropbox.com/s/fpc15vl2oaseo2r/ResNext101_0.pth?dl=0
wget -O ResNext101_1.pth https://www.dropbox.com/s/6ieh20ne21ide0v/ResNext101_1.pth?dl=0
wget -O ResNext101_2.pth https://www.dropbox.com/s/t6mkbaf19cpfg8d/ResNext101_2.pth?dl=0
wget -O ResNext101_3.pth https://www.dropbox.com/s/39clmvpku8w9men/ResNext101_3.pth?dl=0
python3 p1_test.py --test_directory=$1 --output_file=$2