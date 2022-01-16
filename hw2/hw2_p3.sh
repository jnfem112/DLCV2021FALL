#!/bin/bash

:<<!
	python3 p3_train.py \
		--source=svhn \
		--target=mnistm \
		--source_directory=hw2_data/digits/svhn/train/ \
		--source_annotation=hw2_data/digits/svhn/train.csv \
		--target_directory=hw2_data/digits/mnistm/train/ \
		--target_annotation=hw2_data/digits/mnistm/train.csv \
		--test_directory=hw2_data/digits/mnistm/test/ \
		--test_annotation=hw2_data/digits/mnistm/test.csv \
		--model=DaNN \
		--batch_size=64 \
		--learning_rate=0.0001 \
		--lambd=0.1 \
		--epoch=15

	python3 p3_train.py \
		--source=mnistm \
		--target=usps \
		--source_directory=hw2_data/digits/mnistm/train/ \
		--source_annotation=hw2_data/digits/mnistm/train.csv \
		--target_directory=hw2_data/digits/usps/train/ \
		--target_annotation=hw2_data/digits/usps/train.csv \
		--test_directory=hw2_data/digits/usps/test/ \
		--test_annotation=hw2_data/digits/usps/test.csv \
		--model=DaNN \
		--batch_size=64 \
		--learning_rate=0.0001 \
		--lambd=0.1 \
		--epoch=30

	python3 p3_train.py \
		--source=usps \
		--target=svhn \
		--source_directory=hw2_data/digits/usps/train/ \
		--source_annotation=hw2_data/digits/usps/train.csv \
		--target_directory=hw2_data/digits/svhn/train/ \
		--target_annotation=hw2_data/digits/svhn/train.csv \
		--test_directory=hw2_data/digits/svhn/test/ \
		--test_annotation=hw2_data/digits/svhn/test.csv \
		--model=DaNN \
		--batch_size=64 \
		--learning_rate=0.0001 \
		--lambd=0.1 \
		--epoch=15
!

python3 p3_test.py --test_directory=$1 --target=$2 --output_file=$3