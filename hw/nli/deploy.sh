#!/bin/bash
target=/tmp/tf
mkdir $target
rm $target/*.py
rm $target/*.pyc
rm -rf $target/nli-dataset
cp $1 $target
cp nli_dataset.py tools.py $target
cp -r nli-dataset $target

