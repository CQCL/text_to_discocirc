import pickle
from abc import ABC

import numpy as np
import tensorflow as tf
from tensorflow import keras

from network.models.trainer_base_class import DisCoCircTrainerBase


class DisCoCircTrainerIsIn(DisCoCircTrainerBase, ABC):
    def __init__(self, nn_boxes, wire_dimension, is_in_question=None, **kwargs):
        super().__init__(nn_boxes, wire_dimension, **kwargs)
        if is_in_question is None:
            self.is_in_question = self.question_model()
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

    def question_model(self):
        input = keras.Input(shape=(2 * self.wire_dimension))
        output = keras.layers.Dense(self.wire_dimension, activation=tf.nn.relu)(input)
        output = keras.layers.Dense(self.wire_dimension / 2, activation=tf.nn.relu)(output)
        output = keras.layers.Dense(1)(output)
        return keras.Model(inputs=input, outputs=output)

    def get_prediction_result(self, model_output):
        return np.argmax(model_output)

    def get_expected_result(self, given_value):
        return given_value
    
    @tf.function
    def compute_loss(self, context_circuit_model, test):
        person, location = test
        logits = self.call((context_circuit_model, person))
        labels = tf.one_hot(location, logits.shape[0])
        return tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)

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
        # answer_prob = tf.nn.softmax(answer_prob)
        return logits
