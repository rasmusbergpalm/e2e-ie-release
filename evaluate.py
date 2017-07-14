import json
from sys import argv
from typing import Dict, List, Tuple
import numpy as np


def get_counts(expected_objs: List[Dict[str, str]], actual_objs: List[Dict[str, str]], keys) -> Dict[str, List[int]]:
    """
    :return: counts for each key, [correct, incorrect, missing, spurious]
    """
    assert len(expected_objs) == len(actual_objs) > 0
    assert len(keys) > 0

    counts = {k: [0, 0, 0, 0] for k in keys}

    for e, a in zip(expected_objs, actual_objs):
        for k in keys:
            if k in e and k in a:
                if a[k] == e[k]:  # correct
                    counts[k][0] += 1
                else:  # incorrect
                    counts[k][1] += 1
            elif k in e and k not in a:  # missing
                counts[k][2] += 1
            elif k not in e and k in a:  # spurious
                counts[k][3] += 1

    counts['MICRO_AVG'] = np.array(list(counts.values())).sum(axis=0)

    return counts


def metrics_from_counts(counts: List[int]) -> Tuple[float, float, float, float]:
    """
    Computes classifier metrics given counts of correct, incorrect, missing and spurious

    :param counts: A (4,) vector of (correct, incorrect, missing, spurious)
    :return: acc, recall, precision and f1
    """

    eps = 1e-16
    correct, incorrect, missing, spurious = counts

    acc = correct / (correct + incorrect + missing + spurious + eps)
    recall = correct / (correct + incorrect + missing + eps)
    precision = correct / (correct + incorrect + spurious + eps)
    f1 = 2 * (precision * recall) / (recall + precision + eps)

    return acc, recall, precision, f1


def compare(expected_objs: List[Dict[str, str]], actual_objs: List[Dict[str, str]], keys: List[str]) -> Tuple[np.array, Dict[str, Tuple[float, float, float, float]]]:
    counts = get_counts(expected_objs, actual_objs, keys)
    metrics = {k: metrics_from_counts(c) for k, c in counts.items()}
    mean_acc = np.mean([v[0] for v in metrics.values()])
    return mean_acc, metrics


def get_count_matrix(expected_objs: List[Dict[str, str]], actual_objs: List[Dict[str, str]], keys):
    """
    :return: count matrix: N by 4, i.e., all counts for each instance
    """
    assert len(expected_objs) == len(actual_objs) > 0

    counts = np.zeros((len(expected_objs), 4))

    for i, (e, a) in enumerate(zip(expected_objs, actual_objs)):
        for k in keys:
            if k in e and k in a:
                if a[k] == e[k]:  # correct
                    counts[i][0] += 1
                else:  # incorrect
                    counts[i][1] += 1
            elif k in e and k not in a:  # missing
                counts[i][2] += 1
            elif k not in e and k in a:  # spurious
                counts[i][3] += 1

    return counts


def bootstrap(expected, actual1, actual2):
    keys = set()
    for s in expected:
        keys = keys.union(s.keys())

    keys = list(keys)

    acc1, metrics1 = compare(expected, actual1, keys)
    acc2, metrics2 = compare(expected, actual2, keys)

    # establish which system is better on the entire data set
    comparison = np.argmax(np.array([metrics1['MICRO_AVG'], metrics2['MICRO_AVG']]), axis=0)

    # get baselines
    for system_name, metrics in [('SYSTEM1', metrics1), ('SYSTEM2', metrics2)]:
        print(system_name)
        print('*' * 20)
        macro = np.array([v for k, v in metrics.items() if k != 'MICRO_AVG']).mean(axis=0)
        print("MACRO_AVG\t%s" % ('\t'.join(['%.4f' % x for x in macro])))

        for k, v in sorted(metrics.items()):
            print("%s\t%s" % (k, '\t'.join(['%.4f' % x for x in v])))

        print()

    # get counts for both systems
    counts1 = get_count_matrix(expected, actual1, keys)
    counts2 = get_count_matrix(expected, actual2, keys)

    N = 10000  # number of iterations
    D = len(expected)  # number of instances
    K = int(D * 0.75)  # sample size
    diffs = np.zeros(4, dtype=float)

    print("running %s iterations with sample size %s" % (N, K))
    for i in range(N):
        # select instances with replacement
        selection = np.random.choice(D, K)

        # get the counts for those instances in both systems
        selection1 = counts1[selection].sum(axis=0)
        selection2 = counts2[selection].sum(axis=0)

        # compute the metrics for both systems
        selection_metrics1 = metrics_from_counts(selection1)
        selection_metrics2 = metrics_from_counts(selection2)

        winner = np.argmax(np.array([selection_metrics1, selection_metrics2]), axis=0)

        # if the winner on the subset differs from the winner on the entire data
        diffs += (winner != comparison).astype(float)

    print("p-values\t%s" % ('\t'.join(['%.4f' % x for x in diffs / N])))


if __name__ == '__main__':
    """
    usage: python evaluate.py actual1.json actual2.json expected.json
    """
    actual1_fname = argv[1]
    actual2_fname = argv[2]
    expected_fname = argv[3]

    with open(actual1_fname) as f:
        actual1 = json.load(f)
    with open(actual2_fname) as f:
        actual2 = json.load(f)
    with open(expected_fname) as f:
        expected = json.load(f)

    bootstrap(expected, actual1, actual2)
