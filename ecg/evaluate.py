from __future__ import division
from __future__ import print_function
import argparse
import numpy as np
from tqdm import tqdm
import load
import json
import util
import predict
import score


class Evaluator():
    def evaluate(self, ground_truths, probs):
        self.process_ground_truths(ground_truths)
        self.process_probs(probs)
        self.score()


class MultiCategoryEval(Evaluator):
    def __init__(self, classes, decoder=None):
        self.classes = classes
        self.decoder = decoder

    def process_ground_truths(self, ground_truths):
        self.ground_truths = ground_truths

    def process_probs(self, probs):
        if self.decoder:
            raise NotImplementedError()  # TODO: fix
            predictions = np.array(
                [self.decoder.beam_search(probs_indiv)
                 for probs_indiv in tqdm(probs)])
        else:
            predictions = np.argmax(probs, axis=-1)

        predictions = np.tile(
            predictions, (self.ground_truths.shape[0], 1))

        self.predictions = predictions

    def score(self):
        score.score(
            self.ground_truths,
            self.predictions,
            self.classes,
            confusion_table=True,
            report=True)


class BinaryEval(Evaluator):
    def __init__(self, class_int, class_name, threshold):
        self.threshold = threshold
        self.class_int = class_int
        self.class_name = class_name
        self.classes = ['Not ' + class_name, class_name]

    def process_probs(self, probs):
        predictions = probs[:, :, self.class_int]
        mask_as_one = predictions >= self.threshold
        predictions[mask_as_one] = 1
        predictions[~mask_as_one] = 0
        predictions = np.tile(
            predictions, (self.ground_truths.shape[0], 1))
        self.predictions = predictions

    def process_ground_truths(self, ground_truths):
        ground_truths = np.copy(ground_truths)
        class_mask = ground_truths == self.class_int
        ground_truths[class_mask] = 1
        ground_truths[~class_mask] = 0
        self.ground_truths = ground_truths

    def score(self):
        scores = score.score(
            self.ground_truths,
            self.predictions,
            self.classes,
            binary_evaluate=True)
        print(self.class_name, scores, self.threshold)


def evaluate_classes(ground_truths, probs, classes, thresholds=[0.5]):
    for class_int in range(len(classes)):
        for threshold in thresholds:
            evaluator = BinaryEval(class_int, classes[class_int], threshold)
            evaluator.evaluate(ground_truths, probs)


def evaluate_aggregate(ground_truths, probs, classes, decoder=False):
    evaluator = MultiCategoryEval(classes)
    evaluator.evaluate(ground_truths, probs)


def evaluate(args, train_params, test_params):
    x, ground_truths, classes = load.load_test(
            test_params,
            train_params=train_params,
            split=args.split)
    probs = predict.get_ensemble_pred_probs(args.model_paths, x)
    evaluate_aggregate(ground_truths, probs, classes, decoder=args.decode)
    evaluate_classes(ground_truths, probs, classes, np.linspace(0, 1, 5))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("test_config_file", help="path to config file")
    parser.add_argument(
        'model_paths',
        nargs='+',
        help="path to models")
    parser.add_argument("--split", help="train/val", choices=['train', 'test'],
                        default='test')
    parser.add_argument('--decode', action='store_true')
    args = parser.parse_args()
    train_params = util.get_model_params(args.model_paths[0])
    test_params = train_params.copy()
    test_new_params = json.load(open(args.test_config_file, 'r'))
    test_params.update(test_new_params)
    if "label_review" in test_new_params["EVAL_PATH"]:
        assert(args.split == 'test')
    evaluate(args, train_params, test_params)
