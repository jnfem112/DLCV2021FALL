#!/bin/bash

:<<!
	mkdir p1_output/
	python3 p1_train.py
!

python3 p1_test.py --output_directory=$1