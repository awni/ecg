#!/bin/sh
cd ..
python ecg/train.py data configs/default.json $1
