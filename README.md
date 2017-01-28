# ECG - Arrhythmia Detection

#### Before you begin
Here are some commands that setup directory conventions used by later commands.
```bash
# create symlink to data directory
ln -s <path_to_data> data
# create a folder to save models (match with FOLDER_TO_SAVE in configs/)
mkdir saved/
```

#### Training
```bash
# Normal
python ecg/train.py data configs/<choice_of_configs>
# To overfit the training set
python ecg/train.py data configs/<choice_of_configs> --overfit
```

#### Prediction
```bash
python ecg/predict.py data <path_to_saved_model_hdf5> {train/test}
```

#### Evaluation
Before running evaluation on a model, run the prediction script
to generate neccessary prediction files for evaluation.
```bash
python ecg/evaluate.py data <path_to_saved_model_hdf5> {train/test} # use --decode for decoding
```

#### Analyze
Side by side comparison of model parameters and performance.
Models are automatically grouped by their version number, which is to be changed
when the API of the model is not compatible with previous models.
```bash
# Sort results by increasing validation loss (default)
python ecg/analyze.py saved/<version_folder> --version <version_number>
# Sort results by increasing training loss (default)
python ecg/analyze.py saved/<version_folder> --version <version_number> --metric=loss
```