import pickle
from abc import ABC

import numpy as np
import tensorflow as tf

from network.individual_networks_models.individual_networks_trainer_base_class import \
    IndividualNetworksTrainerBase
from network.utils.utils import create_feedforward_network


class IsInIndividualNetworksTrainer(IndividualNetworksTrainerBase, ABC):
    def __init__(self, lexicon=None, wire_dimension=10, is_in_question=None, is_in_hidden_layers=[10], **kwargs):
        super().__init__(lexicon, **kwargs)
        if is_in_question is None:
            self.is_in_question = create_feedforward_network(
                input_dim = 2 * wire_dimension,
                output_dim = 1,
                hidden_layers = is_in_hidden_layers
            )
        else:
            self.is_in_question = is_in_question

    def save_models(self, path):
        kwargs = {
            "nn_boxes": self.nn_boxes,
            "wire_dimension": self.wire_dimension,
            "is_in_question": self.is_in_question
        }
        with open(path, "wb") as f:
            pickle.dump(kwargs, f)

    def get_prediction_result(self, model_output):
        return np.argmax(model_output)

    def get_expected_result(self, given_value):
        return given_value
    
    @tf.function
    def compute_loss(self, context_circuit_model, test):
        person, location = test
        logits = self.call((context_circuit_model, person))
        return tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=[location])

    @tf.function
    def call(self, circ_person):
        circ, person = circ_person
        output_vector = circ(tf.convert_to_tensor([[]]))[0]
        total_wires = output_vector.shape[0] // self.wire_dimension
        person_vector = output_vector[person * self.wire_dimension : (person + 1) * self.wire_dimension]
        logits = []
        for i in range(total_wires):
            location_vector = output_vector[i * self.wire_dimension : (i + 1) * self.wire_dimension]
            logits.append(
                self.is_in_question(
                    tf.expand_dims(tf.concat([person_vector, location_vector], axis=0), axis=0)
                )[0][0]
            )
        logits = tf.convert_to_tensor(logits)
        return logits
