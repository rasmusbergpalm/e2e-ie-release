from unittest import TestCase
import lasagne.layers as L
import theano.tensor as T
import theano
import numpy as np

from layers import BahdanauKeyValueAttentionLayer


class TestBahdanauAttentionLayer(TestCase):
    def test_get_output_for(self):
        keys_var = T.ftensor3()
        values_var = T.ftensor3()
        mask_var = T.fmatrix()
        queries_var = T.ftensor3()

        keys_layer = L.InputLayer((None, None, 3), input_var=keys_var)
        values_layer = L.InputLayer((None, None, 5), input_var=values_var)
        mask_layer = L.InputLayer((None, None), input_var=mask_var)
        queries_layer = L.InputLayer((None, None, 7), input_var=queries_var)

        attention_layer = BahdanauKeyValueAttentionLayer([keys_layer, values_layer, mask_layer, queries_layer], 9)

        attention_outputs = L.get_output(attention_layer)

        fn = theano.function([keys_var, values_var, mask_var, queries_var], attention_outputs, on_unused_input='warn')

        keys = np.random.rand(32, 13, 3).astype(np.float32)
        values = np.random.rand(32, 13, 5).astype(np.float32)
        mask = np.random.rand(32, 13).astype(np.float32)
        queries = np.random.rand(32, 17, 7).astype(np.float32)

        _att = fn(keys, values, mask, queries)

        self.assertEqual((32, 17, 5), _att.shape)
