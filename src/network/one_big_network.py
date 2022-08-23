import os

from network.utils import get_box_name, get_params_dict_from_tf_variables
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras
from discopy.monoidal import Swap
from discopy import Box, Ty
from copy import deepcopy


class MyDenseLayer(keras.layers.Layer):
    @tf.function(jit_compile=True)
    def call(self, input, weights, bias, mask):
        out = tf.einsum("bi,bij->bj", input, weights) + bias
        return tf.where(tf.cast(mask, dtype=tf.bool), tf.nn.relu(out), out)


class NeuralDisCoCirc(keras.Model):
    def __init__(self,
        lexicon=None,
        wire_dimension=20,
        hidden_layers=[50],
        is_in_question=None
    ):
        super().__init__()
        self.wire_dimension = wire_dimension
        self.hidden_layers = hidden_layers
        self.dense_layer = MyDenseLayer()
        self.lexicon = lexicon
        if lexicon:
            self.initialize_lexicon_weights(lexicon)
        self.is_in_question = is_in_question
        if is_in_question is None:
            self.is_in_question = self.question_model()
        self.loss_tracker = keras.metrics.Mean(name="loss")


    def question_model(self):
        input = keras.Input(shape=(2 * self.wire_dimension))
        output = keras.layers.Dense(self.wire_dimension, activation=tf.nn.relu)(input)
        output = keras.layers.Dense(self.wire_dimension / 2, activation=tf.nn.relu)(output)
        output = keras.layers.Dense(1)(output)
        return keras.Model(inputs=input, outputs=output)

    @tf.function(jit_compile=True)
    def call(self, inputs_params):
        inputs, params = inputs_params
        output = inputs
        for i in range(len(params)):
            output = self.dense_layer(output, *params[i])
        return output

    def get_max_width_and_depth(self, diagrams):
        diagrams = [self.diagram_parameters[repr(d)] for d in diagrams]
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
        inputs = [d["input"] for d in diagrams]
        i_sizes = [sum(x.shape[0] for x in i) for i in inputs]
        max_i_sizes = max(i_sizes)
        return max_len, max_layer_size, max_i_sizes

    def batch_diagrams(self, diagrams):
        diagrams = [self.diagram_parameters[repr(d)] for d in diagrams]
        batched_params_for_i = lambda i: self.get_batched_params(diagrams, i)
        layer_params = list(map(batched_params_for_i, range(self.max_depth)))
        inputs = [d["input"] for d in diagrams]
        i_sizes = [sum(x.shape[0] for x in i) for i in inputs]
        tensor_inputs = []
        for i in range(len(inputs)):
            diff = self.max_input_length - i_sizes[i]
            if diff > 0:
                tensor_inputs.append(
                    tf.concat(inputs[i] + [tf.zeros((diff,))],axis=0)
                )
            else:
                tensor_inputs.append(tf.concat(inputs[i], axis=0))
        batched_inputs = tf.stack(tensor_inputs, axis=0)
        return batched_inputs, layer_params

    def get_batched_params(self, diagrams, i):
        layers = [(d["weights"][i], d["biases"][i], d["masks"][i])
                      for d in diagrams]
        current_max_layer_width = self.max_layer_width[i]
        pad_and_make_block_diag_current = lambda layer: self.pad_and_make_block_diag(current_max_layer_width, layer)
        new_layer_params = map(pad_and_make_block_diag_current, layers)
        new_layer_params = list(new_layer_params)
        batched_weights = tf.stack([w[0] for w in new_layer_params], axis=0)
        batched_biases = tf.stack([w[1] for w in new_layer_params], axis=0)
        batched_masks = tf.stack([w[2] for w in new_layer_params], axis=0)
        return batched_weights, batched_biases, batched_masks

    def pad_and_make_block_diag(self, max_layer_width, layer):
        w, b, m = layer[0], layer[1], layer[2]
        block_diag = self._make_block_diag(w)
        diff = (max(max_layer_width[0] - block_diag.shape[0], 0),
                        max(max_layer_width[1] - block_diag.shape[1], 0))
        w_paddings = tf.constant([[0, diff[0]], [0, diff[1]]])
        block_diag = tf.pad(block_diag, w_paddings, "CONSTANT")
        bias = tf.concat(b + [tf.zeros((diff[1],))], axis=0)
        mask = tf.concat(m + [tf.zeros((diff[1],))], axis=0)
        return [block_diag, bias, mask]

    def get_box_layers(self, layers, name):
        weights = [
            self.add_weight(
                shape = (layers[i], layers[i+1]),
                initializer = "glorot_uniform",
                trainable = True,
                name = name + '_weights_' + str(i)
            )
            for i in range(len(layers)-1)
        ]
        biases = [
            self.add_weight(
                shape = (layers[i+1],),
                initializer = "glorot_uniform",
                trainable = True,
                name = name + '_biases_' + str(i)
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
        for i, d in enumerate(diagrams):
            print("\rGetting parameters for diagram: {} of {}".format(i+1, len(diagrams)), end="")
            self.diagram_parameters[repr(d)] = self._get_parameters_from_diagram(d)
        print("\n")
        self.max_depth, self.max_layer_width, self.max_input_length = self.get_max_width_and_depth(self.diagrams)

    def _get_parameters_from_diagram(self, diagram):
        model_weights = []
        model_biases = []
        model_activation_masks = []
        model_input = [self.states[get_box_name(box)] for box in diagram.foliation()[0].boxes]

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
                    layer_weights[i].append(self.lexicon_weights[get_box_name(box)][i])
                    layer_biases[i].append(self.lexicon_biases[get_box_name(box)][i])
                    layer_activation_masks[i].append(tf.ones((self.lexicon_weights[get_box_name(box)][i].shape[1],)))

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
            name = get_box_name(word)
            if input_dim == 0:
                self.states[get_box_name(word)] = self.add_weight(
                    shape = (output_dim,),
                    initializer = "glorot_uniform",
                    trainable = True,
                    name = name + '_states'
                )
            else:
                w, b = self.get_box_layers([input_dim] + self.hidden_layers + [output_dim], name)
                self.lexicon_weights[get_box_name(word)] = w
                self.lexicon_biases[get_box_name(word)] = b
        self.add_swap_weights_and_biases()
    
    def get_lexicon_params_from_saved_variables(self):
        weights = [v for v in self.variables if 'weights' in v.name]
        biases = [v for v in self.variables if 'biases' in v.name]
        states = [v for v in self.variables if 'states' in v.name]
        self.lexicon_weights = get_params_dict_from_tf_variables(weights, '_weights_')
        self.lexicon_biases = get_params_dict_from_tf_variables(biases, '_biases_')
        self.states = get_params_dict_from_tf_variables(states, '_states', is_state=True)
        self.add_swap_weights_and_biases()

    def add_swap_weights_and_biases(self):
        swap = Swap(Ty('n'), Ty('n'))
        e = tf.eye(self.wire_dimension)
        z = tf.zeros([self.wire_dimension, self.wire_dimension])
        a = tf.concat((z, e), axis=1)
        b = tf.concat((e, z), axis=1)
        swap_mat = tf.concat((a, b), axis=0)
        self.lexicon_weights[get_box_name(swap)] = ([swap_mat] + [tf.eye(2 * self.wire_dimension)]
                                                     * len(self.hidden_layers))
        self.lexicon_biases[get_box_name(swap)] = ([tf.zeros((2 * self.wire_dimension,))]
                                                     * (1 + len(self.hidden_layers)))

    def fit(self, dataset, epochs=100, batch_size=32, **kwargs):
        self.diagrams = [data[0] for data in dataset]
        self.tests = [data[1] for data in dataset]
        self.get_parameters_from_diagrams(self.diagrams)
        input_index_dataset = tf.data.Dataset.range(len(dataset))
        input_index_dataset = input_index_dataset.shuffle(len(dataset))
        input_index_dataset = input_index_dataset.batch(batch_size)
        return super().fit(input_index_dataset, epochs=epochs, **kwargs)
    
    def train_step(self, batch_index):
        diagrams = [self.diagrams[int(i)] for i in batch_index]
        tests = [self.tests[int(i)] for i in batch_index]
        with tf.GradientTape() as tape:
            inputs, params = self.batch_diagrams(diagrams)
            outputs = self.call((inputs, params))
            loss = self.compute_loss(outputs, tests)
            grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(
            (grad, weights)
            for (grad, weights) in zip(grads, self.trainable_weights)
            if grad is not None)
        self.loss_tracker.update_state(loss)
        return {
            "loss": self.loss_tracker.result(),
        }
    
    @tf.function(jit_compile=True)
    def compute_loss(self, outputs, tests):
        num_wires = self.max_input_length // self.wire_dimension
        output_wires = tf.split(outputs, num_wires, axis=1)
        tests = np.array(tests).T
        person, location = tests[0], tests[1]
        person = [(person, i) for i, person in enumerate(person)]
        person_vectors = tf.gather_nd(output_wires, person)
        answer_prob = []
        for i in range(num_wires):
            location_vectors = output_wires[i]
            answer_prob.append(tf.squeeze(
                self.is_in_question(
                    tf.concat([person_vectors, location_vectors], axis=1)
                )
            ))
        answer_prob = tf.transpose(answer_prob)
        answer_prob = tf.nn.softmax(answer_prob)
        labels = tf.one_hot(location, answer_prob.shape[1])
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=answer_prob, labels=labels)
        return loss

    def get_probabilities(self, diagrams, tests):
        inputs, params = self.batch_diagrams([diagrams])
        outputs = self.call((inputs, params))
        num_wires = self.max_input_length // self.wire_dimension
        output_wires = tf.split(outputs, num_wires, axis=1)
        tests = np.array(tests).T
        person, location = tests[0], tests[1]
        person = [person]
        person = [(person, i) for i, person in enumerate(person)]
        person_vectors = tf.gather_nd(output_wires, person)
        answer_prob = []
        for i in range(num_wires):
            location_vectors = output_wires[i]
            answer_prob.append(tf.squeeze(
                self.is_in_question(
                    tf.concat([person_vectors, location_vectors], axis=1)
                )
            ))
        answer_prob = tf.transpose(answer_prob)
        answer_prob = tf.nn.softmax(answer_prob)
        return answer_prob

    def get_config(self):
        return {
            "wire_dimension": self.wire_dimension,
            "is_in_question": self.is_in_question,
            "hidden_layers": self.hidden_layers,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    @classmethod
    def load_model(cls, path):
        model = keras.models.load_model(
            path,
            custom_objects = {cls.__name__: cls},
        )
        model.run_eagerly = True
        model.get_lexicon_params_from_saved_variables()
        return model
