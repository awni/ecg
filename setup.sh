#!/bin/bash

if [ -z "$TF" ]
then
    TF=tensorflow
else
    TF=tensorflow-gpu
fi


pip install -r requirements.txt
pip install --upgrade $TF==1.8.0

