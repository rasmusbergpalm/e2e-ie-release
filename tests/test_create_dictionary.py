from unittest import TestCase

from data import create_dictionary


class TestCreate_dictionary(TestCase):
    def test_create_dictionary(self):
        sentences = [
            ["foo", "bar", "baz"],
            ["foo", "bar"],
            ["foo"]
        ]
        expected = {"foo": 0, "bar": 1}
        actual = create_dictionary(sentences, 2, False)
        self.assertEqual(expected, actual)

        expected = {"foo": 0, "bar": 1, "baz": 2}
        actual = create_dictionary(sentences, 3, False)
        self.assertEqual(expected, actual)

        expected = {"foo": 0, "bar": 1, "baz": 2}
        actual = create_dictionary(sentences, None, False)
        self.assertEqual(expected, actual)

        expected = {"foo": 0}
        actual = create_dictionary(sentences, 1, False)
        self.assertEqual(expected, actual)
