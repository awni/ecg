#!/bin/sh
if [[ $# -lt 1 ]] ; then
    echo "Please supply experiment name"
    exit 1
fi
for DEVICE in 0 1 2 3
do
    DEVICE_UTILIZATION=$(nvidia-smi --id=$DEVICE --query-gpu=utilization.gpu --format=csv,nounits,noheader)
    if [ "$DEVICE_UTILIZATION" -gt 2 ]
        then
            echo "GPU $DEVICE not free"
        else
            echo "GPU $DEVICE free"
            export CUDA_VISIBLE_DEVICES=$DEVICE
            python ecg/train.py configs/train.json -e $1 -v 2 > $DEVICE-$1.out 2>&1 &
            echo "Started training script on $DEVICE..."
            
    fi
done
