#!/bin/sh
for DEVICE in 0 1 2 3
do
    DEVICE_UTILIZATION=$(nvidia-smi --id=$DEVICE --query-gpu=utilization.gpu --format=csv,nounits,noheader)
    if [ $(($DEVICE_UTILIZATION + 0)) -gt 2 ]
        then
            echo "GPU $DEVICE not free"
        else
            echo "GPU $DEVICE free"
            export CUDA_VISIBLE_DEVICES=$DEVICE
            python ecg/train.py data configs/default.json $1 > $DEVICE.out 2>&1 &
    fi
echo $OUTPUT
done
