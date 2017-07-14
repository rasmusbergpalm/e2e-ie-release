import json
import random
import subprocess
import time

import keras as K
import numpy as np
from keras.layers import Embedding, Bidirectional, LSTM, TimeDistributed, Dense, Activation
from keras.models import Sequential
from keras.utils.np_utils import to_categorical

import data
import evaluate
from baselines.bilstm.metrics import conlleval
from iob_chunker import IOBChunker


def shuffle(lol, seed):
    '''
    lol :: list of list as input
    seed :: seed the shuffling

    shuffle inplace each list in the same order
    '''
    for l in lol:
        random.seed(seed)
        random.shuffle(l)


if __name__ == '__main__':

    emb_dimension = 128
    nhidden = 128
    seed = 345
    nepochs = 50
    verbose = True
    optimizer = K.optimizers.Adam()

    corpus = "movie"
    words2idx, labels2idx, idx2word, idx2label, keys, train_words, train_labels, valid_words, valid_labels, test_words, test_labels = data.load_mit(corpus)

    expected_objects = data.get_expected_ojects(test_words, test_labels, idx2word, idx2label, keys)
    with open('expected-test-objs.json', 'w') as f:
        json.dump(expected_objects, f)

    vocsize = len(words2idx)
    nclasses = len(labels2idx)
    nsentences = len(train_words)

    np.random.seed(seed)
    random.seed(seed)

    model = Sequential()
    model.add(Embedding(vocsize, emb_dimension))
    model.add(Bidirectional(LSTM(nhidden, activation='sigmoid', return_sequences=True)))
    model.add(LSTM(nhidden, activation='sigmoid', return_sequences=True))
    model.add(TimeDistributed(Dense(units=nclasses)))
    model.add(Activation("softmax"))

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # train with early stopping on validation set
    best_f1 = -np.inf

    groundtruth_test = [list(map(lambda x: idx2label[x], y)) for y in test_labels]
    words_test = [list(map(lambda x: idx2word[x], w)) for w in test_words]
    groundtruth_valid = [list(map(lambda x: idx2label[x], y)) for y in valid_labels]
    words_valid = [list(map(lambda x: idx2word[x], w)) for w in valid_words]

    s = {}
    for e in range(nepochs):
        # shuffle
        shuffle([train_words, train_labels], seed)
        s['ce'] = e
        tic = time.time()

        for i in range(nsentences):
            X = np.asarray([train_words[i]])
            Y = to_categorical(np.asarray(train_labels[i])[:, np.newaxis], nclasses)[np.newaxis, :, :]
            if X.shape[1] == 1:
                continue  # bug with X, Y of len 1
            model.train_on_batch(X, Y)

            if verbose:
                print('[learning] epoch %i >> %2.2f%%' % (e, (i + 1) * 100. / nsentences), 'completed in %.2f (sec) <<\r' % (time.time() - tic))

        # evaluation // back into the real world : idx -> words
        predictions_test = [list(map(lambda x: idx2label[x],
                                     model.predict_on_batch(
                                         np.asarray([x])).argmax(2)[0]))
                            for x in test_words]

        predictions_valid = [list(map(lambda x: idx2label[x],
                                      model.predict_on_batch(
                                          np.asarray([x])).argmax(2)[0]))
                             for x in valid_words]

        # evaluation // compute the accuracy using conlleval.pl
        res_test = conlleval(predictions_test, groundtruth_test, words_test, 'current.test.txt')
        res_valid = conlleval(predictions_valid, groundtruth_valid, words_valid, 'current.valid.txt')

        if res_valid['f1'] > best_f1:
            model.save_weights('best_model.h5', overwrite=True)
            best_f1 = res_valid['f1']
            print('NEW BEST: epoch', e, 'valid F1', res_valid['f1'], 'best test F1', res_test['f1'], ' ' * 20)
            s['vf1'], s['vp'], s['vr'] = res_valid['f1'], res_valid['p'], res_valid['r']
            s['tf1'], s['tp'], s['tr'] = res_test['f1'], res_test['p'], res_test['r']
            s['be'] = e
            subprocess.call(['mv', 'current.test.txt', 'best.test.txt'])
            subprocess.call(['mv', 'current.valid.txt', 'best.valid.txt'])

            actual_objects = [IOBChunker.to_object(w, l, keys) for w, l in zip(words_test, predictions_test)]
            with open('actual-test-objs-%02d.json' % e, 'w') as f:
                json.dump(actual_objects, f)

            acc_test, metrics = evaluate.compare(expected_objects, actual_objects, keys)
            for k, v in metrics.items():
                print("%s\t%.3f\t%.3f\t%.3f\t%.3f" % (k, *v))

        else:
            print('')

    print('BEST RESULT: epoch', s['be'], 'valid F1', s['vf1'], 'best test F1', s['tf1'])
