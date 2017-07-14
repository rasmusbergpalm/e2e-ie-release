from unittest import TestCase

from iob_chunker import IOBChunker


class TestIOBChunker(TestCase):
    def test_chunk(self):
        words = ['from', 'boston', 'to', 'san', 'diego', 'at', 'nine', 'and', 'newark']
        labels = ['O', 'B-from', 'O', 'B-to', 'I-to', 'O', 'O', 'O', 'B-to']
        actual = IOBChunker.chunk(words, labels)
        expected = [('from', 'boston'), ('to', 'san diego'), ('to', 'newark')]
        self.assertEqual(expected, actual)

        expected_object = {'from': 'boston', 'to': 'san diego , newark'}
        actual_object = IOBChunker.to_object(words, labels, ['from', 'to'])
        self.assertEqual(expected_object, actual_object)
