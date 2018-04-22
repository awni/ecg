## Install 

Clone the repository

```
git clone git@github.com:awni/ecg.git
```


Install `virtualenv` with

```
pip install virtualenv
```

Make and activate a new Python 2.7 environment

```
virtualenv -p Python2.7 ecg_env
source ecg_env/bin/activate
```

Install the requirements (this may take a few minutes).

**NB** If you do not have a GPU, change line XX in requirements.txt from 

```
tensorflow-gpu==1.0.1
```

to

```
tensorflow==1.0.1
```

Now run

```
pip install -r requirements.txt
```

## Training

In the repo root direcotry (`ecg`) make a new directory called `saved`.

```
mkdir saved
```

In the same directory download and unpack the data into a folder called `data`.

```
unzip data.zip
```

Then run

```
python ecg/train.py
```

After each epoch the model is saved in
`ecg/saved/default/<experiment_id>/<model_id>.hdf5`.

**NB** this model is only trained on 128 examples. This is far too few to see
good generalization performance, but the code should run and produce a valid
model.

## Testing

After training the model for a few epochs, you can make predictions and
evaluate performance.

```
python ecg/predict.py configs/test_reviewer.json saved/default/<experiment_id>/<model_id>.hdf5
```

And to print some metrics run:

```
python ecg/evaluate.py saved/predictions/<experiment_id>
```

**NB**: These instructions evaluate the model on the training set since we do
not include an independent test set in the sample data.
