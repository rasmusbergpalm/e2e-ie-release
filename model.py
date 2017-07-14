from typing import List

import lasagne
import lasagne.layers as L
import numpy as np
import theano
import theano.tensor as T

from layers import BahdanauKeyValueAttentionLayer, DropoutEmbeddingLayer, DropoutLSTMLayer, RepeatLayer


def e2e_pointer_net(dict_size: int, decoder_names: List[str], go_char_idx: int, eos_char_idx: int):
    def rnn_encoder(x_sym, x_mask):
        name = "Encoder"
        n_layers = 1
        n_units = 128
        emb_size = 128
        rnn = DropoutLSTMLayer

        l_in = L.InputLayer((None, None), input_var=x_sym)
        l_mask = L.InputLayer((None, None), input_var=x_mask)
        l_emb = DropoutEmbeddingLayer(l_in, dict_size, emb_size, name=name + '.Embedding', dropout=0.25)
        l_onehot = L.EmbeddingLayer(l_in, dict_size, dict_size, W=np.eye(dict_size, dtype='float32'), name=name + '.OneHot')
        l_onehot.params[l_onehot.W].remove('trainable')

        l_enc_forwards = rnn(l_emb, num_units=n_units, mask_input=l_mask, name=name + '.0.Forward')
        l_enc_backwards = rnn(l_emb, num_units=n_units, mask_input=l_mask, backwards=True, name=name + '.0.Backward')
        l_enc = L.ConcatLayer([l_enc_forwards, l_enc_backwards], axis=2)

        for i in range(n_layers - 1):
            l_enc = rnn(l_enc, num_units=n_units, mask_input=l_mask, name="%s.%d.Forward" % (name, i + 1), dropout=0.25)

        return l_onehot, l_enc

    def rnn_decoder(l_input_one_hot, l_encoder_hid, encoder_mask, out_sym, out_mask, out_go_sym, name="Decoder"):
        n_layers = 1
        n_units = 256
        n_attention_units = 256
        emb_size = 256
        rnn = DropoutLSTMLayer

        l_go_out = L.InputLayer((None, None), input_var=out_go_sym)
        l_out_mask = L.InputLayer((None, None), input_var=out_mask)
        l_in_mask = L.InputLayer((None, None), input_var=encoder_mask)

        l_emb = L.EmbeddingLayer(l_go_out, dict_size, emb_size, name=name + '.Embedding')

        last_hid_encoded = L.SliceLayer(rnn(l_encoder_hid, num_units=n_units, mask_input=l_in_mask, name=name + '.Summarizer', dropout=0.25), indices=-1, axis=1)
        encoder_last_hid_repeat = RepeatLayer(last_hid_encoded, n=T.shape(out_go_sym)[1], axis=1)

        l_dec = L.ConcatLayer([l_emb, encoder_last_hid_repeat], axis=2)
        for i in range(n_layers):
            l_dec = rnn(l_dec, num_units=n_units, mask_input=l_out_mask, name="%s.%d.Forward" % (name, i), learn_init=True, dropout=0.25)

        l_attention = BahdanauKeyValueAttentionLayer([l_encoder_hid, l_input_one_hot, l_in_mask, l_dec], n_attention_units, name=name + '.Attention')  # (bs, seq_out, dict)
        l_out = L.ReshapeLayer(l_attention, (-1, [2]))

        out_random = L.get_output(l_out, deterministic=False)  # (batch * seq_out) x dict
        out_deterministic = L.get_output(l_out, deterministic=True)  # (batch * seq_out) x dict
        params = L.get_all_params([l_out], trainable=True)

        rcrossentropy = T.nnet.categorical_crossentropy(out_random + 1e-8, out_sym.flatten())  # (batch * seq) x 1
        crossentropy = T.reshape(rcrossentropy, (bs, -1))  # batch x seq
        loss = T.sum(out_mask * crossentropy) / T.sum(out_mask)  # scalar

        argmax = T.argmax(T.reshape(out_deterministic, (bs, -1, dict_size)), axis=-1)  # batch x seq x 1

        return {'loss': loss, 'argmax': argmax, 'params': params}

    learning_rate = T.scalar(name='learning_rate')
    in_sym = T.imatrix()
    bs = in_sym.shape[0]
    in_mask_sym = T.fmatrix()

    l_input_one_hot, l_encoded_hid = rnn_encoder(in_sym, in_mask_sym)

    out_syms = []
    out_go_syms = []
    out_mask_syms = []
    decoded = []
    for i, name in enumerate(decoder_names):
        out_syms.append(T.imatrix())
        out_go_syms.append(T.imatrix())
        out_mask_syms.append(T.fmatrix())

        decoded.append(rnn_decoder(l_input_one_hot, l_encoded_hid, in_mask_sym, out_syms[i], out_mask_syms[i], out_go_syms[i], name=name))

    all_params = list(set.union(*[set(d['params']) for d in decoded]))
    all_params = sorted(all_params, key=lambda x: x.name)
    loss = sum(d['loss'] for d in decoded)
    argmaxes = [d['argmax'] for d in decoded]

    print("Trainable Model Parameters")
    total = 0
    for param in all_params:
        shp = param.get_value().shape
        total += np.product(shp)
        print(param, shp)
    print("Total %d" % total)

    all_grads = [T.clip(g, -3., 3.) for g in T.grad(loss, all_params)]
    all_grads = lasagne.updates.total_norm_constraint(all_grads, 3.)

    updates = lasagne.updates.adam(all_grads, all_params, learning_rate=learning_rate)

    print("Compiling functions...")
    train_tfn = theano.function([in_sym, in_mask_sym] + out_syms + out_mask_syms + out_go_syms + [learning_rate], loss, updates=updates)
    argmax_tfn = theano.function([in_sym, in_mask_sym] + out_go_syms + out_mask_syms, argmaxes)

    def prepend_go(out):
        return (np.append(go_char_idx * np.ones((out.shape[0], 1)), out, axis=1)[:, :-1]).astype(np.int32)

    def train_fn(input, input_mask, outputs, output_masks, learning_rate):
        output_gos = [prepend_go(out) for out in outputs]
        return train_tfn(input, input_mask, *outputs, *output_masks, *output_gos, learning_rate)

    def test_fn(test_batch):
        input, input_mask = test_batch[:2]
        output_masks = test_batch[2 + len(decoder_names):]
        max_len = max([np.max(np.argwhere(m > 0)[:, 1]) for m in output_masks]) + 1

        bs = input.shape[0]
        outputs = [go_char_idx * np.ones((bs, 1)).astype(np.int32) for i in range(len(decoder_names))]
        iters = 0
        while iters < max_len and not all([np.all(out[:, -1] == eos_char_idx) for out in outputs]):
            out_masks = [(outputs[i] != eos_char_idx).astype(np.float32) for i in range(len(decoder_names))]
            argmaxes = argmax_tfn(input, input_mask, *outputs, *out_masks)

            for i, argmax in enumerate(argmaxes):
                outputs[i] = np.append(outputs[i], argmax[:, [-1]], axis=1).astype(np.int32)

            iters += 1

        outputs = [out[:, 1:] for out in outputs]  # clip go char input

        return outputs

    def load(name):
        with np.load(name) as data:
            values = data['values']
        for p, v in zip(all_params, values):
            p.set_value(v)

    def save(name):
        values = [p.get_value().astype(np.float32) for p in all_params]
        np.savez_compressed(name, values=values)

    return train_fn, test_fn, save, load
