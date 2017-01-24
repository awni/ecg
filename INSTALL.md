## Installation Instruction for running on the deep cluster with Python 2 and GPU enabled Tensorflow
```
# *NB* if you are on AFS you may not have enough space in your home directory
# for the environment. I recommend putting it in scratch or somewhere where 
# you have a few GB of space.
$HOME/.local/bin/virtualenv -p /usr/bin/python2.7 ecg_env
source ecg_env/bin/activate # add to .bashrc.user

pip install -r path_to/requirements.txt

# install tensorflow for GPU
export TF_BINARY_URL= # Get 0.12.1 from TF site
pip install --upgrade $TF_BINARY_URL

## Add below to .bashrc.user
# for cuda 
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib:/usr/local/cuda-8.0/lib64
# for cuda nvcc
export PATH=$PATH:/usr/local/cuda-8.0/bin:
```