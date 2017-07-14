import json
import sys

import data
import evaluate
from model import e2e_pointer_net

sys.setrecursionlimit(10000)


def main(dataset):
    batch_size = 64

    keys = dataset.keys
    expected_objects = dataset.expected_objects
    train_func, test_func, save_func, load_func = e2e_pointer_net(dataset.dict_size(), keys, dataset.go_idx, dataset.eos_idx)

    print("Loading model...")
    load_func('best.npz')

    print("Testing...")
    actual_objects = []
    for val_batch in dataset.val_batch(batch_size):
        val_actual_outputs = test_func(val_batch)
        actual_objects.extend(dataset.to_objs(val_actual_outputs))

    with open('actual-test-objs.json', 'w') as f:
        json.dump(actual_objects, f)

    mean_acc, metrics = evaluate.compare(expected_objects, actual_objects, keys)
    for k, v in metrics.items():
        print("%s\t%.3f\t%.3f\t%.3f\t%.3f" % (k, *v))


if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] not in {"atis", "movie", "restaurant"}:
        print("Usage: python test.py atis|movie|restaurant")
        exit(1)

    dstring = sys.argv[1]

    if dstring == "atis":
        dataset = data.atis(use_validation=False)
    elif dstring == "movie":
        dataset = data.movie(use_validation=False)
    else:
        dataset = data.restaurant(use_validation=False)

    main(dataset)
