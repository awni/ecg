import evaluate
import util
import json
import load
import numpy as np

args = util.get_object_from_dict(data_path="./data/label_review")
params = json.load(open('./configs/test.json', 'r'))

params["epi_ext"] = "_rev1.episodes.json"
dl1 = load.load_train(args, params)
ground_truth = np.argmax(dl1.y_test, axis=-1)

params["epi_ext"] = "_rev0.episodes.json"
dl2 = load.load_train(args, params)
predictions = np.argmax(dl2.y_test, axis=-1)

evaluate.compute_scores(ground_truth, predictions, dl1.classes)
