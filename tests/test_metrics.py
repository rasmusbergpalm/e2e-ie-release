from unittest import TestCase

from evaluate import get_counts


class TestMetrics(TestCase):
    def test_metrics(self):
        expected = [
            {'foo': 'yes',
             'bar': 'no'},

            {'foo': 'yes',
             'bar': 'no'},

            {'foo': 'maybe'}
        ]

        actual = [
            {'foo': 'yes',
             'bar': 'no'},

            {'foo': 'yes',
             'bar': 'yes'},

            {'baz': 'maybe',
             'bar': 'oops'}
        ]

        keys = ['foo', 'bar', 'baz']
        counts = get_counts(expected, actual, keys)

        self.assertEqual(set(keys).union({"MICRO_AVG"}), counts.keys())

        # correct, incorrect, missing, spurious
        self.assertEqual(counts['foo'], [2, 0, 1, 0])
        self.assertEqual(counts['bar'], [1, 1, 0, 1])
        self.assertEqual(counts['baz'], [0, 0, 0, 1])
