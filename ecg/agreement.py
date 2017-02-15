import evaluate
import util
import json
import load
import itertools
import numpy as np

NUM_LABELLERS = 3
classes = None


def get_labels_of_labeler(args, params, index):
    global classes
    assert(index >= 0 and index < NUM_LABELLERS)
    params["epi_ext"] = "_rev" + str(index) + ".episodes.json"
    dl = load.load_train(args, params)
    if classes is None:
        classes = dl.classes
    else:
        assert(classes == dl.classes)
    return np.argmax(dl.y_test, axis=-1)


def agreement():
    args = util.get_object_from_dict(data_path="./data/label_review")
    params = json.load(open('./configs/test.json', 'r'))
    all_labels = [get_labels_of_labeler(args, params, i) for
                  i in range(NUM_LABELLERS)]
    for pair in itertools.combinations(range(NUM_LABELLERS), 2):
        print("Agreement between " + str(pair))
        evaluate.compute_scores(
            all_labels[pair[0]],
            all_labels[pair[1]],
            classes,
            confusion_table=False,
            plot=False)


if __name__ == '__main__':
    agreement()

