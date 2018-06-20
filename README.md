## Install 

Clone the repository

```
git clone git@github.com:awni/ecg.git
```

If you don't have `virtualenv`, install it with

```
pip install virtualenv
```

Make and activate a new Python 2.7 environment

```
virtualenv -p python2.7 ecg_env
source ecg_env/bin/activate
```

Install the requirements (this may take a few minutes).

For CPU only support run
```
./setup.sh
```

To install with GPU support run
```
env TF=gpu ./setup.sh
```

## Training

In the repo root direcotry (`ecg`) make a new directory called `saved`.

```
mkdir saved
```

To train a model use the following command, replacing `path_to_config.json`
with an actual config:

```
python ecg/train.py path_to_config.json
```

Note that after each epoch the model is saved in
`ecg/saved/<experiment_id>/<timestamp>/<model_id>.hdf5`.

For an actual example of how to run this code on a real dataset, you can follow
the instructions in the cinc17 [README](examples/cinc17/README.md). This will
walk through downloading the Physionet 2017 challenge dataset and training and
evaluating a model.

## Testing

After training the model for a few epochs, you can make predictions with.

```
python ecg/predict.py <dataset>.json <model>.hdf5
```

replacing `<dataset>` with an actual path to the dataset and `<model>` with the
path to the model.
