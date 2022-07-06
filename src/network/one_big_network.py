from tkinter import HIDDEN
import tensorflow as tf
from tensorflow import keras
from discopy import Box


class MyDenseLayer(keras.layers.Layer):
    def call(self, x, kernel, bias):
        return tf.einsum("bi,bij->bj", x, kernel) + bias


class NeuralDisCoCirc(keras.Model):
    def __init__(self, lexicon, wire_dimension=20, hidden_layers=[50]):
        super().__init__()
        self.wire_dimension = wire_dimension
        self.hidden_layers = hidden_layers
        self.dense_layer = MyDenseLayer()
        self.initialize_lexicon_weights(lexicon)

    @tf.function(jit_compile=True)
    def call(self, x, weights, biases):
        weights = tf.transpose(weights, perm=[1,0,2,3])
        biases = tf.transpose(biases, perm=[1,0,2])
        output = x
        for i in range(tf.shape(weights)[0]):
            output = self.dense_layer(output, weights[i], biases[i])
        return output

    # TODO: batch the diagrams to make use of MyDenseLayer()
    def batch_diagrams(self, diagrams):
        pass

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
            self.diagram_parameters[d] = self._get_parameters_from_diagram(d)


    def _get_parameters_from_diagram(self, diagram):
        model_weights = []
        model_biases = []
        model_activation_masks = []

        inputs = []
        for box in diagram.foliation()[0].boxes:
            inputs.append(self.states[box])
        model_input = tf.concat(inputs, axis=0)

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

            weight_matrices = [self._make_block_diag(w) for w in layer_weights]
            model_weights += weight_matrices

            bias_vectors = [tf.concat(b, axis=0) for b in layer_biases]
            model_biases += bias_vectors

            activation_masks = [tf.concat(a, axis=0) for a in layer_activation_masks]
            model_activation_masks += activation_masks

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
