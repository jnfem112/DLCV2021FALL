#!/bin/bash

:<<!
    python3 -W ignore p1_train.py
    python3 -W ignore test_testcase.py --load=prototypical_network.pth --test_csv=hw4_data/mini/val.csv --test_data_dir=hw4_data/mini/val/ --testcase_csv=hw4_data/mini/val_testcase.csv --output_csv=prediction.csv
    python3 -W ignore eval.py prediction.csv hw4_data/mini/val_testcase_gt.csv
!

python3 -W ignore test_testcase.py --load=prototypical_network.pth --test_csv=$1 --test_data_dir=$2 --testcase_csv=$3 --output_csv=$4