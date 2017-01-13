

```
gpu=0
config=mitdb_config.json
env CUDA_VISIBLE_DEVICES=$gpu python ecg/train.py --config=$config
```

To view results run:
```
port=8888
log_dir=<directory_of_saved_models>
tensorboard --port $port --log_dir $log_dir
```
