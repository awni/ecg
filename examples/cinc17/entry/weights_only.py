
import keras
import sys

mpath = sys.argv[1]
model = keras.models.load_model(mpath)
model.save_weights("model.hdf5")
