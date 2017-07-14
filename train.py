import json
import sys

import numpy as np

import data
import evaluate
from model import e2e_pointer_net

sys.setrecursionlimit(10000)


def main(dataset):
    val_interval = 100
    num_updates_total = 100000
    batch_size = 64
    learning_rate = 1e-3

    keys = dataset.keys
    expected_objects = dataset.expected_objects
    train_func, test_func, save_func, load_func = e2e_pointer_net(dataset.dict_size(), keys, dataset.go_idx, dataset.eos_idx)

    losses = []
    best_acc = -1
    print("Training...")
    try:
        for num_updates in range(num_updates_total):
            train_batch = dataset.train_batch(batch_size)

            source, source_mask = train_batch[:2]
            outputs = train_batch[2:2 + len(keys)]
            output_masks = train_batch[2 + len(keys):]

            loss = train_func(source, source_mask, outputs, output_masks, learning_rate)
            losses += [loss]
            if num_updates % val_interval == 0:
                learning_rate *= 0.95

                avg_loss = np.array(losses[-val_interval:]).mean()
                print("%05d/%05d loss: %f" % (num_updates, num_updates_total, avg_loss))

                actual_objects = []
                for val_batch in dataset.val_batch(batch_size):
                    val_actual_outputs = test_func(val_batch)
                    actual_objects.extend(dataset.to_objs(val_actual_outputs))

                with open('actual-val-objs-%02d.json' % num_updates, 'w') as f:
                    json.dump(actual_objects, f)

                mean_acc, metrics = evaluate.compare(expected_objects, actual_objects, keys)

                if mean_acc > best_acc:
                    best_acc = mean_acc
                    save_func('best.npz')
                    print("New best: %f" % mean_acc)

                for k, v in metrics.items():
                    print("%s\t%.3f\t%.3f\t%.3f\t%.3f" % (k, *v))

        return best_acc

    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] not in {"atis", "movie", "restaurant"}:
        print("Usage: python train.py atis|movie|restaurant")
        exit(1)

    dstring = sys.argv[1]

    if dstring == "atis":
        dataset = data.atis(use_validation=True)
    elif dstring == "movie":
        dataset = data.movie(use_validation=True)
    else:
        dataset = data.restaurant(use_validation=True)

    main(dataset)
