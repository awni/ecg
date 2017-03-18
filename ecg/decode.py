from __future__ import division
from __future__ import print_function
from builtins import range
from builtins import object
import numpy as np
import math
import collections


class LM(object):

    def __init__(self, y_train, vocab_size, order):
        SOME_LARGE_NEGATIVE_NUMBER = -1000
        self.order = order
        counts = [collections.Counter() for _ in range(vocab_size)]
        labels = np.argmax(y_train, axis=-1)

        for label in labels:
            for i in range(order, len(label) + 1):
                ngram = tuple(label[i - order:i])
                counts[label[i - 1]][ngram] += 1

        ngrams = []
        for counter in counts:
            log_tot = math.log(sum(t for _, t in counter.most_common()))
            scores = {ngram: math.log(t) - log_tot
                      for ngram, t in counter.most_common()}
            ngrams.append(scores)

        self.ngrams = ngrams
        self.default = SOME_LARGE_NEGATIVE_NUMBER

    def score_ngram(self, ngram):
        ngram = tuple(ngram)
        scores = self.ngrams[ngram[-1]]
        return scores.get(ngram, self.default)

class Decoder:

    def __init__(self, y_train, vocab_size,
                 order=3, beam_size=4, lm_weight=2.0):
        self.lm = LM(y_train, vocab_size, order)
        self.beam_size = beam_size
        self.lm_weight = lm_weight

    def beam_search(self, probs):
        lm = self.lm
        beam_size = self.beam_size
        lm_weight = self.lm_weight

        probs = np.log(probs.squeeze())
        (T, S) = probs.shape
        beam = [([], 0.0)]
        for t in range(T):
            new_beam = []
            for candidate, score in beam:
                for c in range(S):
                    new_cand = list(candidate)  # copy
                    new_cand.append(c)
                    new_score = score + probs[t, c]
                    if lm_weight is not None and len(new_cand) >= lm.order:
                        ngram = new_cand[-lm.order:]
                        new_score += lm_weight * lm.score_ngram(ngram)
                    new_beam.append((new_cand, new_score))

            # Sort and trim the beam
            beam = sorted(new_beam, key=lambda x: x[1], reverse=True)
            beam = beam[:beam_size]

        return beam[0][0]
