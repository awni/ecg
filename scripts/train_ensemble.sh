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
            screen -d -m bash -c "python ecg/train.py data configs/default.json $1 > $DEVICE.out 2>&1"
            echo "Started training script on $DEVICE..."
    fi
done
