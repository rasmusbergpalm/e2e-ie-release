import hashlib
import json
import os
import pickle
import random
import urllib.request
import zipfile

import numpy as np

from iob_chunker import IOBChunker

UNK = "<UNK>"


def atis(use_validation):
    return MultiOutput(*load_atis(), use_validation=use_validation)


def restaurant(use_validation):
    return MultiOutput(*load_mit("restaurant"), use_validation=use_validation)


def movie(use_validation):
    return MultiOutput(*load_mit("movie"), use_validation=use_validation)


class MultiOutput:
    eos_char = "#"
    go_char = "!"

    def __init__(self, words2idx, labels2idx, idx2word, idx2label, keys, train_words, train_labels, valid_words, valid_labels, test_words, test_labels, delimiter=" ", use_validation=True):
        self.delimiter = delimiter
        self.keys = keys

        if use_validation:
            test_words = valid_words
            test_labels = valid_labels

        self.expected_objects = get_expected_ojects(test_words, test_labels, idx2word, idx2label, keys)
        with open('expected-%s-objs.json' % ('val' if use_validation else 'test'), 'w') as f:
            json.dump(self.expected_objects, f)

        test_in, test_outs, train_in, train_outs = self.to_multi_outputs(keys, words2idx, labels2idx, train_words, train_labels, test_words, test_labels)

        max_idx = max(words2idx.values())
        for ew in {self.eos_char, self.go_char}:
            words2idx[ew] = max_idx + 1
            max_idx += 1

        self.words2idx = words2idx
        self.idx2words = dict((v, k) for k, v in self.words2idx.items())

        self.eos_idx = self.words2idx[self.eos_char]
        self.go_idx = self.words2idx[self.go_char]

        train_in, test_in = self.pad_to_max_len(train_in, test_in)

        train_outs, test_outs = zip(*[self.pad_to_max_len(train_out, test_out) for train_out, test_out in zip(train_outs, test_outs)])

        train_input, train_in_mask = self.mask(train_in)
        test_input, test_in_mask = self.mask(test_in)

        train_outputs, train_out_masks = zip(*[self.mask(train_out) for train_out in train_outs])
        test_outputs, test_out_masks = zip(*[self.mask(test_out) for test_out in test_outs])

        self.train_samples = list(zip(train_input, train_in_mask, *train_outputs, *train_out_masks))
        self.test_samples = list(zip(test_input, test_in_mask, *test_outputs, *test_out_masks))

    def to_multi_outputs(self, keys, words2idx, labels2idx, train_in, train_labels, test_in, test_labels):
        # tokens that can be part of output but are not guaranteed to be part of input can be added here.
        # They are pre-pended to every input sentence.
        extra_tokens = [","]
        extra_idx = []
        for token in extra_tokens:
            if token not in words2idx:
                words2idx[token] = len(words2idx)
            extra_idx.append(words2idx[token])

        idx2words = dict((v, k) for k, v in words2idx.items())
        idx2labels = dict((v, k) for k, v in labels2idx.items())

        def extract(sentences, labels, key):
            O = []

            for sw, sl in zip(sentences, labels):
                w = [idx2words[_w] for _w in sw]
                l = [idx2labels[_l] for _l in sl]
                chunks = IOBChunker.chunk(w, l)
                d = [v for k, v in chunks if k == key]

                joined = " , ".join(d)
                idxs = [words2idx[w] for w in joined.split()]

                O.append(idxs)

            return O

        train_in = [list(s) for s in train_in]
        test_in = [list(s) for s in test_in]
        train_outs = []
        test_outs = []
        for key in keys:
            train_outs.append(extract(train_in, train_labels, key))
            test_outs.append(extract(test_in, test_labels, key))
        train_in = [extra_idx + s for s in train_in]
        test_in = [extra_idx + s for s in test_in]
        assert len(keys) == len(train_outs) == len(test_outs)
        return test_in, test_outs, train_in, train_outs

    def pad_to_max_len(self, train_seq, test_seq):
        seqmax = len(max(train_seq + test_seq, key=len))
        train_seq = [x + [self.eos_idx] * (seqmax - len(x) + 1) for x in train_seq]
        test_seq = [x + [self.eos_idx] * (seqmax - len(x) + 1) for x in test_seq]
        return train_seq, test_seq

    def mask(self, idx: list):
        idx = np.array(idx).astype(np.int32)
        mask = np.ones(idx.shape)
        mask[idx == self.eos_idx] = 0.
        mask = np.append(np.ones((mask.shape[0], 1)), mask, axis=1)[:, :-1]
        return idx, mask.astype(np.float32)

    def prepend_go(self, output_idx):
        shifted = np.copy(output_idx)
        shifted = np.insert(shifted, 0, self.go_idx * np.ones((1, shifted.shape[0])), axis=1)
        return shifted[:, :-1].astype(np.int32)

    def sample(self, samples, batch_size):
        sample = random.sample(samples, batch_size)
        return list(map(np.array, zip(*sample)))

    def train_batch(self, batch_size):
        return self.sample(self.train_samples, batch_size)

    def val_batch(self, batch_size):
        l = len(self.test_samples)
        for ndx in range(0, l, batch_size):
            yield list(map(np.array, zip(*self.test_samples[ndx:min(ndx + batch_size, l)])))

    def output_to_str(self, outputs):
        """
        :param output: np.array (bs, num_tokens)
        :return: the chars as a list of strings, until the first mask char "#"
        """
        return [self.delimiter.join([self.idx2words[c] for c in output]).partition(self.eos_char)[0].strip()
                for output in outputs.tolist()]

    def dict_size(self):
        return len(self.words2idx)

    def to_objs(self, outputs):
        strings = dict()
        for i, k in enumerate(self.keys):
            strings[k] = self.output_to_str(outputs[i])

        objects = []
        for i in range(outputs[0].shape[0]):
            obj = {k: strings[k][i] for k in self.keys if strings[k][i]}
            objects.append(obj)

        return objects


def get_expected_ojects(words_idx, labels_idx, idx2word, idx2label, keys):
    labels = [list(map(lambda x: idx2label[x], y)) for y in labels_idx]
    words = [list(map(lambda x: idx2word[x], w)) for w in words_idx]

    return [IOBChunker.to_object(w, l, keys) for w, l in zip(words, labels)]


def read_conll_format(fname):
    with open(fname) as f:
        lines = f.readlines()

    left = []
    right = []
    l = []
    r = []
    for line in lines:
        if not line.strip():  # empty line
            left.append(l)
            right.append(r)
            l = []
            r = []
        else:
            _l, _r = line.strip().split()
            l.append(_l)
            r.append(_r)

    return left, right


def ensure_exists(url: str, fpath: str):
    if not os.path.exists(fpath):
        download_fname = '/tmp/' + hashlib.md5(url.encode("UTF-8")).hexdigest()
        if not os.path.exists(download_fname):
            print("Downloading " + url)
            urllib.request.urlretrieve(url, download_fname)

        with zipfile.ZipFile(download_fname, 'r') as zipf:
            zipf.extractall('/tmp/')

    assert os.path.exists(fpath)


def create_dictionary(sentences, max_size, add_unk):
    counts = dict()
    for s in sentences:
        for w in s:
            if w not in counts:
                counts[w] = 1
            else:
                counts[w] += 1

    counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    if max_size is not None and len(counts) > max_size:
        counts = counts[:max_size]

    if add_unk:
        counts.append((UNK, 0))

    word2idx = dict()
    for item in counts:
        word2idx[item[0]] = len(word2idx)

    return word2idx


def map_to_idx(sentences, mapping):
    return [list(map(lambda x: mapping[x] if x in mapping else mapping[UNK], s)) for s in sentences]


def get_keys(label2idx):
    keys = []
    for k in label2idx.keys():
        if k == "O":
            continue
        else:
            position, value = k.split("-")
            if value not in keys:
                keys.append(value)

    return keys


def load_atis():
    fname = "/tmp/atis.pkl"
    if not os.path.exists(fname):
        print("Downloading data...")
        urllib.request.urlretrieve("https://www.dropbox.com/s/eytsgfvii0ctyhs/atis.pkl?dl=1", fname)

    with open(fname, 'rb') as f:
        train_set, test_set, dic = pickle.load(f, encoding='latin1')

    words2idx = dic['words2idx']
    labels2idx = dic['labels2idx']

    train_words, train_ne, train_labels = train_set
    test_words, test_ne, test_labels = test_set

    valid_frac = 0.1
    n_valid = round(len(train_words) * valid_frac)

    valid_words = train_words[:n_valid]
    valid_labels = train_labels[:n_valid]
    train_words = train_words[n_valid:]
    train_labels = train_labels[n_valid:]

    idx2label = dict((k, v) for v, k in labels2idx.items())
    idx2word = dict((k, v) for v, k in words2idx.items())

    keys = [
        'fromloc.city_name',
        'toloc.city_name',
        'airline_name',
        'cost_relative',
        'depart_time.period_of_day',
        'depart_time.time',
        'depart_time.time_relative',
        'depart_date.day_name',
        'depart_date.day_number',
        'depart_date.month_name',
    ]

    return words2idx, labels2idx, idx2word, idx2label, keys, train_words, train_labels, valid_words, valid_labels, test_words, test_labels


def load_mit(corpus):
    assert corpus == "movie" or corpus == "restaurant"

    mit_url = "https://www.dropbox.com/s/yix1146gbzd71ma/mit-corpus.zip?dl=1"
    mit_folder = '/tmp/mit-corpus/'
    ensure_exists(mit_url, mit_folder)

    train_labels, train_words = read_conll_format(mit_folder + corpus + '-train.bio')
    test_labels, test_words = read_conll_format(mit_folder + corpus + '-test.bio')

    words2idx = create_dictionary(train_words, None, True)
    labels2idx = create_dictionary(train_labels, None, False)

    train_words = map_to_idx(train_words, words2idx)
    train_labels = map_to_idx(train_labels, labels2idx)

    test_words = map_to_idx(test_words, words2idx)
    test_labels = map_to_idx(test_labels, labels2idx)

    idx2label = dict((k, v) for v, k in labels2idx.items())
    idx2word = dict((k, v) for v, k in words2idx.items())

    keys = get_keys(labels2idx)

    valid_frac = 0.1
    n_valid = round(len(train_words) * valid_frac)

    valid_words = train_words[:n_valid]
    valid_labels = train_labels[:n_valid]
    train_words = train_words[n_valid:]
    train_labels = train_labels[n_valid:]

    return words2idx, labels2idx, idx2word, idx2label, keys, train_words, train_labels, valid_words, valid_labels, test_words, test_labels
