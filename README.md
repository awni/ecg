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
python ecg/train.py configs/train.json # use --overfit for overfitting
```

#### Evaluation
```bash
python ecg/evaluate.py configs/val.json <list_of_paths_to_saved_model_hdf5> # use --decode for decoding
```

#### Analyze
Side by side comparison of model parameters and performance.
Models are automatically grouped by their version number, which is to be changed
when the API of the model is not compatible with previous models.
```bash
# Sort results by increasing validation loss (default)
python ecg/analyze.py saved/<version_folder> --version <version_number>
# Sort results by increasing training loss (default)
python ecg/analyze.py saved/<version_folder> --version <version_number> --metric=train_loss
```