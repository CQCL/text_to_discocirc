import pickle
import tensorflow as tf
from tensorflow import keras

from network.trainer_base_class import DisCoCircTrainerBase

class DisCoCircTrainerIsIn(DisCoCircTrainerBase):
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
    
    @tf.function
    def compute_loss(self, context_circuit_model, test):
        person, location = test
        answer_prob = self.call((context_circuit_model, person))
        labels = tf.one_hot(location, answer_prob.shape[0])
        return tf.nn.softmax_cross_entropy_with_logits(logits=answer_prob, labels=labels)

    @tf.function
    def call(self, circ_person):
        circ, person = circ_person
        output_vector = circ(tf.convert_to_tensor([[]]))[0]
        total_wires = output_vector.shape[0] // self.wire_dimension
        person_vector = output_vector[person * self.wire_dimension : (person + 1) * self.wire_dimension]
        answer_prob = []
        for i in range(total_wires):
            location_vector = output_vector[i * self.wire_dimension : (i + 1) * self.wire_dimension]
            answer_prob.append(
                self.is_in_question(
                    tf.expand_dims(tf.concat([person_vector, location_vector], axis=0), axis=0)
                )[0][0]
            )
        answer_prob = tf.convert_to_tensor(answer_prob)
        answer_prob = tf.nn.softmax(answer_prob)
        return answer_prob
