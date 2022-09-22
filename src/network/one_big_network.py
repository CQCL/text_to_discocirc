import os

os.environ["CUDA_VISIBLE_DEVICES"]="0"

from abc import ABC, abstractmethod
import numpy as np
import tensorflow as tf
from tensorflow import keras
from discopy.monoidal import Swap
from discopy import Box, Ty
from copy import deepcopy

from network.utils import get_box_name, get_params_dict_from_tf_variables


class MyDenseLayer(keras.layers.Layer):
    @tf.function(jit_compile=True)
    def call(self, input, weights, bias, mask):
        out = tf.einsum("bi,bij->bj", input, weights) + bias
        return tf.where(tf.cast(mask, dtype=tf.bool), tf.nn.relu(out), out)


class NeuralDisCoCirc(keras.Model, ABC):
    def __init__(self,
        lexicon=None,
        wire_dimension=20,
        hidden_layers=[50],
    ):
        super().__init__()
        self.wire_dimension = wire_dimension
        self.hidden_layers = hidden_layers
        self.dense_layer = MyDenseLayer()
        self.lexicon = lexicon
        if lexicon:
            self.initialize_lexicon_weights(lexicon)
        self.loss_tracker = keras.metrics.Mean(name="loss")

    @tf.function(jit_compile=True)
    def call(self, params):
        input, weight, bias, mask = params
        output = input
        for weight, bias, mask in zip(weight, bias, mask):
            output = self.dense_layer(output, weight, bias, mask)
        return output
    
    def get_max_input_length(self, diagram_parameters):
        inputs = [d["input"] for d in diagram_parameters.values()]
        input_length = [sum(x.shape[0] for x in i) for i in inputs]
        return max(input_length)

    def get_max_layer_widths(self, diagram_parameters):
        max_layer_widths = []
        for d in diagram_parameters.values():
            for i in range(len(d['weights'])):
                if len(max_layer_widths) <= i:
                    max_layer_widths.append((0,0))
                max_layer_widths[i] = (
                    max(max_layer_widths[i][0], sum(w.shape[0] for w in d['weights'][i])),
                    max(max_layer_widths[i][1], sum(w.shape[1] for w in d['weights'][i]))
                )
        return max_layer_widths
    
    def extend_depth_of_diagram_parameters(self, diagram_parameters):
        max_depth = max([len(d['weights']) for d in diagram_parameters.values()])
        for d in diagram_parameters.values():
            diff = max_depth - len(d['weights'])
            if diff > 0:
                last_layer_width = sum(w.shape[1] for w in d['weights'][-1])
                d['weights'].extend([[tf.eye(last_layer_width)] for _ in range(diff)])
                d["biases"].extend([[tf.zeros((last_layer_width,))] for _ in range(diff)])
                d["masks"].extend([[tf.zeros((last_layer_width,))] for _ in range(diff)])
        return diagram_parameters

    def batch_diagrams(self, diagrams):
        diagrams = [self.diagram_parameters[repr(d)] for d in diagrams]
        inputs = tf.stack(
            [tf.concat(d['input'], axis=0) for d in diagrams],
            axis = 0
        )
        weights = []
        biases = []
        masks = []
        for i in range(len(diagrams[0]['weights'])):
            weights.append(tf.stack(
                [self._make_block_diag(d['weights'][i]) for d in diagrams],
                axis = 0
            ))
            biases.append(tf.stack(
                [tf.concat(d['biases'][i], axis=0) for d in diagrams],
                axis = 0
            ))
            masks.append(tf.stack(
                [tf.concat(d['masks'][i], axis=0) for d in diagrams],
                axis = 0
            ))
        return inputs, weights, biases, masks
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
        # Manually keeping track of the shape of a to make the method jit compilable
        #TODO: remove this if jit_compile doesn't work
        a_shape = (a.shape[0], a.shape[1])
        for i in range(1, len(weights)):
            b = weights[i]
            a_temp = tf.concat((a, tf.zeros((a_shape[0], b.shape[1]))), axis=1)
            b_temp = tf.concat((tf.zeros((b.shape[0], a_shape[1])), b), axis=1)
            a_shape = (a_shape[0] + b.shape[0], a_shape[1] + b.shape[1])
            a = tf.concat((a_temp, b_temp), axis=0)
        return a

    def get_parameters_from_diagrams(self, diagrams):
        self.diagram_parameters = {}
        for i, d in enumerate(diagrams):
            print("\rGetting parameters for diagram: {} of {}".format(i+1, len(diagrams)), end="")
            self.diagram_parameters[repr(d)] = self._get_parameters_from_diagram(d)
        print("\n")
        self.diagram_parameters = self.extend_depth_of_diagram_parameters(self.diagram_parameters)
        self.max_layer_widths = self.get_max_layer_widths(self.diagram_parameters)
        self.max_input_length = self.get_max_input_length(self.diagram_parameters) 
        self.add_padding_to_parameters(self.diagram_parameters, self.max_layer_widths, self.max_input_length)

    def _get_parameters_from_diagram(self, diagram):
        model_weights = []
        model_biases = []
        model_activation_masks = []
        model_input = [self.states[get_box_name(box)] for box in diagram.foliation()[0].boxes]

        for fol in diagram.foliation()[1:]:
            layer_weights, layer_biases, layer_activation_masks = self.get_parameters_from_foliation(fol)
            model_weights += layer_weights
            model_biases += layer_biases
            model_activation_masks += layer_activation_masks

        return {'input': model_input,
                'weights': model_weights,
                'biases': model_biases,
                'masks': model_activation_masks,
                }

    def get_parameters_from_foliation(self, foliation):
        weights = [[]]
        biases = [[]]
        activation_masks = [[]]
        if any(type(e) == Box for e in foliation.boxes):
            weights = [[] for _ in range(1 + len(self.hidden_layers))]
            biases = [[] for _ in range(1 + len(self.hidden_layers))]
            activation_masks = [[] for _ in range(1 + len(self.hidden_layers))]

        wires_traversed = 0
        for left, box, right in foliation.layers:
            if len(left) > wires_traversed: # new identity wires are introduced
                weights, biases, activation_masks = self.add_id_params_to_layer(
                    len(left) - wires_traversed, weights, biases, activation_masks)
            for i in range(len(weights)):
                weights[i].append(self.lexicon_weights[get_box_name(box)][i])
                biases[i].append(self.lexicon_biases[get_box_name(box)][i])
                activation_masks[i].append(tf.ones((self.lexicon_weights[get_box_name(box)][i].shape[1],)))
            wires_traversed = len(left) + len(box.cod)
        if right: # identity wires on the right that were not traversered
            weights, biases, activation_masks = self.add_id_params_to_layer(
                len(right), weights, biases, activation_masks)
        return weights, biases, activation_masks
    
    def add_id_params_to_layer(self, num_id_wires, weights, biases, activation_masks):
        for i in range(len(weights)):
            weights[i] += ([tf.eye(self.wire_dimension)] * num_id_wires)
            biases[i] += ([tf.zeros((self.wire_dimension,))] * num_id_wires)
            activation_masks[i] += ([tf.zeros((self.wire_dimension,))] * num_id_wires)
        return weights, biases, activation_masks

    def add_padding_to_parameters(self, diagram_parameters, max_layer_widths, max_input_length):
        for d in diagram_parameters.values():
            input_size = sum(x.shape[0] for x in d['input'])
            if input_size < max_input_length:
                diff = max_input_length - input_size
                d['input'].append(tf.zeros((diff,)))
            for i in range(len(d['weights'])):
                diff_0 = max_layer_widths[i][0] - sum([x.shape[0] for x in d['weights'][i]])
                diff_1 = max_layer_widths[i][1] - sum([x.shape[1] for x in d['weights'][i]])
                d['weights'][i].append(tf.zeros((diff_0, diff_1)))
                d['biases'][i].append(tf.zeros((diff_1,)))
                d['masks'][i].append(tf.zeros((diff_1,)))
        return diagram_parameters

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
            batched_params = self.batch_diagrams(diagrams)
            outputs = self.call(batched_params)
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
    
    @abstractmethod
    def compute_loss(self, outputs, tests):
        pass

    def get_config(self):
        return {
            "wire_dimension": self.wire_dimension,
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
