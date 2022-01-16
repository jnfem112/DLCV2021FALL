#!/bin/bash

:<<!
    python3 -W ignore p2_pretrain.py --learning_rate=0.0003 --epoch=100
    python3 -W ignore p2_train.py
!

python3 -W ignore p2_test.py --test_annotation=$1 --test_directory=$2 --output_file=$3