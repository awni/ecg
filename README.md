Install dependencies for running on the deep cluster with Python 3 and GPU enabled Tensorflow

```
wget https://bootstrap.pypa.io/get-pip.py
python3 get-pip.py --user
$HOME/.local/bin/pip3 install virtualenv --user

# *NB* if you are on AFS you may not have enough space in your home directory
# for the environment. I recommend putting it in scratch or somewhere where 
# you have a few GB of space.
$HOME/.local/bin/virtualenv ecg_env
source ecg_env/bin/activate # add to .bashrc.user


pip install -r path_to/requirements.txt

# install tensorflow for GPU
export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-0.12.1-cp34-cp34m-linux_x86_64.whl
pip install --upgrade $TF_BINARY_URL


## Add below to .bashrc.user
# for cuda 
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib:/usr/local/cuda-8.0/lib64

# for cuda nvcc
export PATH=$PATH:/usr/local/cuda-8.0/bin:
```

Run with
```
gpu=0
config=mitdb_config.json
env CUDA_VISIBLE_DEVICES=$gpu python train.py --config=$config
```

To view results run:
```
port=8888
log_dir=<directory_of_saved_models>
tensorboard --port $port --log_dir $log_dir
```
