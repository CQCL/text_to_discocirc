import pickle

import numpy as np
import tensorflow as tf
from tensorflow import keras

from network.models.trainer_base_class import DisCoCircTrainerBase


class DisCoCircTrainerAddScaledLogits(DisCoCircTrainerBase):
    def __init__(self, nn_boxes, wire_dimension, lexicon=None, relevance_question=None, is_in_question=None, vocab_dict=None, **kwargs):
        super().__init__(nn_boxes, wire_dimension, lexicon=lexicon, **kwargs)

        if vocab_dict is None:
            vocab_dict = {}
            for i, v in enumerate(lexicon):
                vocab_dict[v.name] = i

        self.vocab_dict = vocab_dict

        if is_in_question is None:
            self.is_in_question = self.question_model(len(self.vocab_dict))
        else:
            self.is_in_question = is_in_question

        if relevance_question is None:
            self.relevance_question = self.question_model(1)
        else:
            self.relevance_question = relevance_question

    def save_models(self, path):
        kwargs = {
            "nn_boxes": self.nn_boxes,
            "wire_dimension": self.wire_dimension,
            "is_in_question": self.is_in_question,
            "relevance_question": self.relevance_question,
            "vocab_dict": self.vocab_dict
        }
        with open(path, "wb") as f:
            pickle.dump(kwargs, f)

    def question_model(self, output_size):
        input = keras.Input(shape=(2 * self.wire_dimension))
        output = keras.layers.Dense(self.wire_dimension, activation=tf.nn.relu)(input)
        output = keras.layers.Dense(self.wire_dimension / 2, activation=tf.nn.relu)(output)
        output = keras.layers.Dense(output_size)(output)
        return keras.Model(inputs=input, outputs=output)

    def get_prediction_result(self, call_result):
        return np.argmax(call_result)

    def get_expected_result(self, given_value):
        return self.vocab_dict[given_value]

    @tf.function
    def compute_loss(self, context_circuit_model, test):
        person, location = test
        answer_prob = self.call((context_circuit_model, person))
        labels = tf.one_hot(self.vocab_dict[location], answer_prob.shape[0])
        return tf.nn.softmax_cross_entropy_with_logits(logits=answer_prob, labels=labels)

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
        relevances = tf.nn.softmax(relevances)

        for i in range(total_wires):
            location_vector = output_vector[i * self.wire_dimension : (i + 1) * self.wire_dimension]

            answer = self.is_in_question(
                    tf.expand_dims(tf.concat([person_vector, location_vector], axis=0), axis=0)
                )[0]

            logit = tf.convert_to_tensor(answer)
            logit_sum = tf.math.add(tf.math.multiply(logit, relevances[i]), logit_sum)

        return logit_sum
