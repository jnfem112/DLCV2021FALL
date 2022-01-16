#!/bin/bash

:<<!
	mkdir p2_output/
	python3 p2_train.py
!

python3 p2_test.py --output_directory=$1