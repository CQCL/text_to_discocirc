import pickle

import numpy as np
import tensorflow as tf

from network.individual_networks_models.individual_networks_trainer_base_class import \
    IndividualNetworksTrainerBase
from network.utils.utils import create_feedforward_network


class AddScaledLogitsIndividualNetworksTrainer(IndividualNetworksTrainerBase):
    def __init__(self,
                 nn_boxes,
                 wire_dimension,
                 lexicon=None,
                 relevance_question=None,
                 is_in_question=None,
                 vocab_dict=None,
                 relevance_hidden_layers=[10, 10],
                 is_in_hidden_layers=[10, 10],
                 softmax_relevancies=False,
                 softmax_logits=False,
                 **kwargs):
        super().__init__(nn_boxes, wire_dimension, lexicon=lexicon, **kwargs)

        self.softmax_relevancies = softmax_relevancies
        self.softmax_logits = softmax_logits

        if vocab_dict is None:
            vocab_dict = {}
            for i, v in enumerate(lexicon):
                vocab_dict[v.name] = i

        self.vocab_dict = vocab_dict

        if is_in_question is None:
            self.is_in_question = create_feedforward_network(
                input_dim = 2 * wire_dimension,
                output_dim = len(self.vocab_dict),
                hidden_layers = is_in_hidden_layers
            )
        else:
            self.is_in_question = is_in_question

        if relevance_question is None:
            self.relevance_question = create_feedforward_network(
                input_dim = 2 * wire_dimension,
                output_dim = 1,
                hidden_layers = relevance_hidden_layers
            )
        else:
            self.relevance_question = relevance_question

    def save_models(self, path):
        kwargs = {
            "nn_boxes": self.nn_boxes,
            "wire_dimension": self.wire_dimension,
            "is_in_question": self.is_in_question,
            "relevance_question": self.relevance_question,
            "vocab_dict": self.vocab_dict,
        }
        with open(path, "wb") as f:
            pickle.dump(kwargs, f)

    def get_prediction_result(self, call_result):
        return np.argmax(call_result)

    def get_expected_result(self, given_value):
        return self.vocab_dict[given_value]

    @tf.function
    def compute_loss(self, context_circuit_model, test):
        person, location = test
        answer_prob = self.call((context_circuit_model, person))
        return tf.nn.sparse_softmax_cross_entropy_with_logits(logits=answer_prob, labels=self.vocab_dict[location])

    @tf.function
    def call(self, circ_person):
        circ, person = circ_person
        output_vector = circ(tf.convert_to_tensor([[]]))[0]
        total_wires = output_vector.shape[0] // self.wire_dimension
        person_vector = output_vector[person * self.wire_dimension : (person + 1) * self.wire_dimension]
        logit_sum = tf.zeros(len(self.vocab_dict))
        relevances = []
        for i in range(total_wires):
            location_vector = output_vector[i * self.wire_dimension : (i + 1) * self.wire_dimension]

            relevance = self.relevance_question(
                    tf.expand_dims(tf.concat([person_vector, location_vector], axis=0), axis=0)
                )[0][0]
            relevances.append(relevance)

        relevances = tf.convert_to_tensor(relevances)
        if self.softmax_relevancies:
            relevances = tf.nn.softmax(relevances)

        for i in range(total_wires):
            location_vector = output_vector[i * self.wire_dimension : (i + 1) * self.wire_dimension]

            answer = self.is_in_question(
                    tf.expand_dims(tf.concat([person_vector, location_vector], axis=0), axis=0)
                )[0]

            logit = tf.convert_to_tensor(answer)
            if self.softmax_logits:
                logit = tf.nn.softmax(logit)
            logit_sum = tf.math.add(tf.math.multiply(logit, relevances[i]), logit_sum)

        return logit_sum
