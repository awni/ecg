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
import decode


class Evaluator():
    def __init__(self, scorer):
        self.scorer = scorer
        self.score_params = {}

    def _seq_to_set_gt(self):
        self.set_gt = self._seq_to_set(self.seq_gt)

    def _seq_to_set_preds(self):
        self.set_preds = self._seq_to_set(self.seq_preds)

    def _repeat_seq_preds(self):
        self.seq_preds = np.tile(
            self.seq_preds, (self.seq_gt.shape[0], 1))

    def _flat_seq_gt(self):
        self.seq_gt = self.seq_gt.reshape((-1, self.seq_gt.shape[-1]))

    def score(self, gt, preds):
        self.scorer.score(
            gt,
            preds,
            **self.score_params)

    def evaluate(self, ground_truths, probs, metric='seq'):
        assert(metric in ['set', 'seq'])
        self._to_seq_gt(ground_truths)
        self._to_seq_preds(probs)
        self._repeat_seq_preds()
        self._flat_seq_gt()
        if metric == 'seq':
            self.score(
                self.seq_gt.ravel(), self.seq_preds.ravel())
        else:
            self._seq_to_set_gt()
            self._seq_to_set_preds()
            self.score(
                self.set_gt, self.set_preds)


class MulticlassEval(Evaluator):
    def __init__(self, scorer, classes, decoder=None):
        Evaluator.__init__(self, scorer)
        self.classes = classes
        self.decoder = decoder
        self.score_params = {
            'classes': classes
        }

    def _seq_to_set(self, arr):
        labels = [set(
            np.unique(record_labels).tolist()) for record_labels in arr]
        from sklearn.preprocessing import MultiLabelBinarizer
        mlb = MultiLabelBinarizer()
        return mlb.fit_transform(labels)

    def _to_seq_gt(self, ground_truths):
        self.seq_gt = ground_truths

    def _to_seq_preds(self, probs):
        if self.decoder is not None:
            predictions = np.array(
                [self.decoder.beam_search(probs_indiv)
                 for probs_indiv in tqdm(probs)])
        else:
            predictions = np.argmax(probs, axis=-1)

        self.seq_preds = predictions


class BinaryEval(Evaluator):
    def __init__(self, scorer, class_int, class_name, threshold):
        Evaluator.__init__(self, scorer)
        self.threshold = threshold
        self.class_int = class_int
        self.class_name = class_name
        self.score_params = {
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

    def _to_seq_gt(self, ground_truths):
        ground_truths = np.copy(ground_truths)
        class_mask = ground_truths == self.class_int
        ground_truths[class_mask] = 1
        ground_truths[~class_mask] = 0
        self.seq_gt = ground_truths

    def _to_seq_preds(self, probs):
        probs = np.copy(probs)
        predictions = probs[:, :, self.class_int]
        mask_as_one = predictions >= self.threshold
        predictions[mask_as_one] = 1
        predictions[~mask_as_one] = 0
        self.seq_preds = predictions


def evaluate_binary(
        ground_truths, probs, classes, thresholds, metric, model_title):
    scorer = score.BinaryScorer(model_title=model_title, metric=metric)
    for class_int in tqdm(range(len(classes))):
        for threshold in thresholds:
            evaluator = BinaryEval(
                scorer, class_int, classes[class_int], threshold)
            evaluator.evaluate(ground_truths, probs, metric=metric)
    scorer.display_scores()


def evaluate_multiclass(
        ground_truths, probs, classes, metric, model_title, decoder=None):
    scorer = score.MulticlassScorer(metric=metric, model_title=model_title)
    evaluator = MulticlassEval(scorer, classes, decoder=decoder)
    evaluator.evaluate(ground_truths, probs, metric=metric)
    scorer.display_scores()


def evaluate_all(
        gt, probs, classes,
        model_title='', thresholds=[0.5], decoder=None):
    for metric in ['seq', 'set']:
        evaluate_multiclass(
            gt, probs, classes, metric, model_title, decoder=decoder)
        """
        evaluate_binary(
            gt, probs, classes, thresholds, metric, model_title)
        """


def evaluate(args, train_params, test_params):
    x, gt, processor, loader = load.load_test(
            test_params,
            train_params=train_params,
            split=args.split)
    probs = predict.get_ensemble_pred_probs(args.model_paths, x)
    thresholds = np.linspace(0, 1, 6, endpoint=False)
    decoder = decode.Decoder(loader.y_train, len(processor.classes)) \
        if args.decode else None
    evaluate_all(
        gt, probs, processor.classes, model_title=', '.join(args.model_paths),
        thresholds=thresholds, decoder=decoder)


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
    if "label_review" in test_new_params["data_path"]:
        assert(args.split == 'test')
    evaluate(args, train_params, test_params)
