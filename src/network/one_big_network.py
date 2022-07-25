from tkinter import HIDDEN
import tensorflow as tf
from tensorflow import keras
from discopy.monoidal import Swap
from discopy import Box, Ty
from copy import deepcopy


class MyDenseLayer(keras.layers.Layer):
    def call(self, input, weights, bias, mask):
        out = tf.einsum("bi,bij->bj", input, weights) + bias
        return tf.where(tf.cast(mask, dtype=tf.bool), tf.nn.relu(out), out)


class NeuralDisCoCirc(keras.Model):
    def __init__(self, lexicon, wire_dimension=20, hidden_layers=[50]):
        super().__init__()
        self.wire_dimension = wire_dimension
        self.hidden_layers = hidden_layers
        self.dense_layer = MyDenseLayer()
        self.initialize_lexicon_weights(lexicon)

    @tf.function(jit_compile=True)
    def call(self, diagrams):
        inputs, weights, biases, masks = self.batch_diagrams(diagrams)
        output = inputs
        for i in range(len(weights)):
            output = self.dense_layer(output, weights[i], biases[i], masks[i])
        return output

    def batch_diagrams(self, diagrams):
        diagrams = [deepcopy(self.diagram_parameters[repr(d)]) for d in diagrams]
        max_len = max([len(d["weights"]) for d in diagrams])

        max_layer_size = [(0, 0) for _ in range(max_len)]
        for d in diagrams:
            weights = d["weights"]
            diff = max_len - len(weights)

            if diff > 0:
                final_shape = sum(w.shape[1] for w in weights[-1])
                d["weights"] += ([[tf.eye(final_shape)]] * diff)
                d["biases"] += ([[tf.zeros((final_shape,))]] * diff)
                d["masks"] += ([[tf.zeros((final_shape,))]] * diff)

            for i in range(max_len):
                max_layer_size[i] =  (max(max_layer_size[i][0],
                                          sum(w.shape[0] for w in weights[i])),
                                      max(max_layer_size[i][1],
                                           sum(w.shape[1] for w in weights[i])))

        # FINE UP TO HERE!

        batched_weights = []
        batched_biases = []
        batched_masks = []
        for i in range(max_len):
            layers = [(d["weights"][i], d["biases"][i], d["masks"][i])
                      for d in diagrams]

            layer_weights = []
            layer_biases = []
            layer_masks = []
            for w, b, m in layers:
                block_diag = self._make_block_diag(w)
                diff = (max(max_layer_size[i][0] - block_diag.shape[0], 0),
                        max(max_layer_size[i][1] - block_diag.shape[1], 0))

                w_paddings = tf.constant([[0, diff[0]], [0, diff[1]]])
                block_diag = tf.pad(block_diag, w_paddings, "CONSTANT")
                layer_weights.append(block_diag)

                bias = tf.concat(b + [tf.zeros((diff[1],))], axis=0)
                layer_biases.append(bias)

                mask = tf.concat(m + [tf.zeros((diff[1],))], axis=0)
                layer_masks.append(mask)

            batched_weights.append(tf.stack(layer_weights, axis=0))
            batched_biases.append(tf.stack(layer_biases, axis=0))
            batched_masks.append(tf.stack(layer_masks, axis=0))

        inputs = [d["input"] for d in diagrams]
        i_sizes = [sum(x.shape[0] for x in i) for i in inputs]
        max_len = max(i_sizes)

        tensor_inputs = []
        for i in range(len(inputs)):
            diff = max_len - i_sizes[i]
            if diff > 0:
                tensor_inputs.append(tf.concat(inputs[i] + [tf.zeros((diff,))],
                                            axis=0))
            else:
                tensor_inputs.append(tf.concat(inputs[i], axis=0))

        batched_inputs = tf.stack(tensor_inputs, axis=0)

        return batched_inputs, batched_weights, batched_biases, batched_masks

    def get_box_layers(self, layers):
        weights = [
            self.add_weight(
                shape = (layers[i], layers[i+1]),
                initializer = "glorot_uniform",
                trainable = True
            )
            for i in range(len(layers)-1)
        ]

        biases = [
            self.add_weight(
                shape = (layers[i+1],),
                initializer = "glorot_uniform",
                trainable = True
            )
            for i in range(len(layers)-1)
        ]

        return weights, biases

    @staticmethod
    def _make_block_diag(weights):
        a = weights[0]
        for i in range(1, len(weights)):
            b = weights[i]
            a_temp = tf.concat((a, tf.zeros((a.shape[0], b.shape[1]))), axis=1)
            b_temp = tf.concat((tf.zeros((b.shape[0], a.shape[1])), b), axis=1)
            a = tf.concat((a_temp, b_temp), axis=0)

        return a

    def get_parameters_from_diagrams(self, diagrams):
        self.diagram_parameters = {}

        for d in diagrams:
            self.diagram_parameters[repr(d)] = self._get_parameters_from_diagram(d)


    def _get_parameters_from_diagram(self, diagram):
        model_weights = []
        model_biases = []
        model_activation_masks = []
        model_input = [self.states[box] for box in diagram.foliation()[0].boxes]

        for fol in diagram.foliation()[1:]:
            layer_weights = [[]]
            layer_biases = [[]]
            layer_activation_masks = [[]]

            in_idx = 0
            out_idx = 0
            disco_layers = fol.layers
            inps = fol.dom

            if any(type(e) == Box for e in fol.boxes):
                layer_weights = [[] for _ in range(1 + len(self.hidden_layers))]
                layer_biases = [[] for _ in range(1 + len(self.hidden_layers))]
                layer_activation_masks = [[] for _ in range(1 + len(self.hidden_layers))]

            for d_layer in disco_layers:
                left, box, right = d_layer

                n_wires = len(inps[:in_idx+len(left)-out_idx])
                idx = len(inps[:in_idx])

                if idx < n_wires:
                    for i in range(len(layer_weights)):
                        layer_weights[i] += ([tf.eye(self.wire_dimension)] * (n_wires - in_idx))
                        layer_biases[i] += ([tf.zeros((self.wire_dimension,))] * (n_wires - in_idx))
                        layer_activation_masks[i] += ([tf.zeros((self.wire_dimension,))] * (n_wires - in_idx))

                for i in range(len(layer_weights)):
                    layer_weights[i].append(self.lexicon_weights[box][i])
                    layer_biases[i].append(self.lexicon_biases[box][i])
                    layer_activation_masks[i].append(tf.ones((self.lexicon_weights[box][i].shape[1],)))

                in_idx = len(left) - out_idx + len(box.dom)
                out_idx = len(left @ box.cod)

            if right:
                for i in range(len(layer_weights)):
                    layer_weights[i] += ([tf.eye(self.wire_dimension)] * len(right))
                    layer_biases[i] += ([tf.zeros((self.wire_dimension,))] * len(right))
                    layer_activation_masks[i] += ([tf.zeros((self.wire_dimension,))] * len(right))

            model_weights += layer_weights
            model_biases += layer_biases
            model_activation_masks += layer_activation_masks

        return {"input": model_input,
                "weights": model_weights,
                "biases": model_biases,
                "masks": model_activation_masks,
                }

    def initialize_lexicon_weights(self, lexicon):
        self.lexicon_weights = {}
        self.lexicon_biases = {}
        self.states = {}
        for word in lexicon:
            input_dim = len(word.dom) * self.wire_dimension
            output_dim = len(word.cod) * self.wire_dimension
            if input_dim == 0:
                self.states[word] = self.add_weight(
                    shape = (output_dim,),
                    initializer = "glorot_uniform",
                    trainable = True
                )
            else:
                w, b = self.get_box_layers([input_dim] + self.hidden_layers + [output_dim])
                self.lexicon_weights[word] = w
                self.lexicon_biases[word] = b

        swap = Swap(Ty('n'), Ty('n'))
        e = tf.eye(self.wire_dimension)
        z = tf.zeros([self.wire_dimension, self.wire_dimension])
        a = tf.concat((z, e), axis=1)
        b = tf.concat((e, z), axis=1)
        swap_mat = tf.concat((a, b), axis=0)

        self.lexicon_weights[swap] = ([swap_mat] + [tf.eye(2 * self.wire_dimension)]
                                      * len(self.hidden_layers))
        self.lexicon_biases[swap] = ([tf.zeros((2 * self.wire_dimension,))]
                                     * (1 + len(self.hidden_layers)))


# import pickle
# with open('data/task_vocab_dicts/en_qa1_train.p', 'rb') as f:
#     vocab = pickle.load(f)

# neural_discocirc = NeuralDisCoCirc(20, vocab)

# total_dim = 500
# depth = 100
# batch = 32
# x = tf.random.normal(shape=(batch, total_dim))
# weights = tf.random.normal(shape=(batch, depth, total_dim, total_dim))
# biases = tf.random.normal(shape=(batch, depth, total_dim))

# neural_discocirc(x, weights, biases)

# from time import time
# start_time = time()
# for i in range(1000):
#     neural_discocirc(x, weights, biases)
# end_time = time()
# print('total time', end_time - start_time, 'seconds')
