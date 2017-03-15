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
    def _seq_to_set_gt(self):
        gt = self.seq_gt.reshape((-1, self.seq_gt.shape[-1]))
        set_gt = self._seq_to_set(gt)
        self.set_gt = set_gt

    def _seq_to_set_preds(self):
        set_preds = self._seq_to_set(self.seq_preds)
        self.set_preds = set_preds

    def evaluate(self, ground_truths, probs):
        self._to_gt(ground_truths)
        self._to_preds(probs)
        self._seq_to_set_gt()
        self._seq_to_set_preds()
        self.seq_score()
        self.set_score()

    def seq_score(self):
        score.seq_score(
            self.seq_gt,
            self.seq_preds,
            self.classes,
            **self.score_params)

    def set_score(self):
        score.set_score(
            self.set_gt,
            self.set_preds,
            self.classes,
            **self.score_params)


class MultiCategoryEval(Evaluator):
    def __init__(self, classes, decoder=None):
        self.classes = classes
        self.decoder = decoder
        self.score_params = {
            'confusion_table': True,
            'report': True
        }

    def _seq_to_set(self, arr):
        labels = [set(
            np.unique(record_labels).tolist()) for record_labels in arr]
        from sklearn.preprocessing import MultiLabelBinarizer
        mlb = MultiLabelBinarizer()
        return mlb.fit_transform(labels)

    def _to_gt(self, ground_truths):
        self.seq_gt = ground_truths

    def _to_preds(self, probs):
        if self.decoder:
            raise NotImplementedError()  # TODO: fix
            predictions = np.array(
                [self.decoder.beam_search(probs_indiv)
                 for probs_indiv in tqdm(probs)])
        else:
            predictions = np.argmax(probs, axis=-1)

        predictions = np.tile(
            predictions, (self.seq_gt.shape[0], 1))

        self.seq_preds = predictions


class BinaryEval(Evaluator):
    def __init__(self, class_int, class_name, threshold):
        self.threshold = threshold
        self.class_int = class_int
        self.class_name = class_name
        self.classes = ['Not ' + class_name, class_name]
        self.score_params = {
            'is_binary': True,
            'class_name': class_name,
            'threshold': threshold
        }

    def _seq_to_set(self, arr):
        set_records = []
        for record_labels in arr.astype('int'):
            unique = set(np.unique(record_labels))
            unique.discard(0)
            set_records.append(list(unique))
        from sklearn import preprocessing
        lb = preprocessing.MultiLabelBinarizer(classes=[1])
        return lb.fit_transform(set_records)

    def _to_gt(self, ground_truths):
        ground_truths = np.copy(ground_truths)
        class_mask = ground_truths == self.class_int
        ground_truths[class_mask] = 1
        ground_truths[~class_mask] = 0
        self.seq_gt = ground_truths

    def _to_preds(self, probs):
        probs = np.copy(probs)
        predictions = probs[:, :, self.class_int]
        mask_as_one = predictions >= self.threshold
        predictions[mask_as_one] = 1
        predictions[~mask_as_one] = 0
        predictions = np.tile(
            predictions, (self.seq_gt.shape[0], 1))
        self.seq_preds = predictions


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
    evaluate_classes(
        ground_truths, probs, classes, np.linspace(0, 1, 3, endpoint=False))

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
