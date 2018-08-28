
# The Physionet 2017 Challenge 

Before following this guide first follow the setup instructions in the top-level
[README](../../README.md).

These instructions go through the training and evaluation of a model on the
[Physionet 2017 challenge](https://www.physionet.org/challenge/2017/) dataset.

## Data

To download and build the datasets run:

```
./setup.sh
```

## Training

Change directory to the repo root directory (`ecg`) and run

```
python ecg/train.py examples/cinc17/config.json -e cinc17
```

## Evaluation

The test dataset for the Physionet 2017 challenge is hidden and maintained by
the challenge organizers. To evaluate on this dataset requires packaging and
submitting the code, dependencies and model to a test server. In general you
will need to be familiar with the instructions on the challenge
[website](https://www.physionet.org/challenge/2017/), but we have included some
scripts to make this as simple as possible.

First change the file in `entry/AUTHORS.txt` to be your name and institution.

Next, from the `entry` directory, run

```
./prepare-entry.sh <path_to_model>
```

The model path should be in
`<path_to_repo>/ecg/saved/cinc17/<timestamp>/<best_model>.hdf5`. The dev set
loss is the first number in the model file name, so the best model (as
evaluated by dev set loss) is the model with the smallest first number in its
name.

Note that this script is quite slow since every time the model is run on a
record it has to be reloaded. Once complete, a zip file should be created in
`entry/entry/entry.zip`. This is the submission.
