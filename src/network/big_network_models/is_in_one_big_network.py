import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from tensorflow import keras

from network.big_network_models.one_big_network import NeuralDisCoCirc


class TrainerIsIn(NeuralDisCoCirc):
    def __init__(self,
        is_in_question=None,
        **kwargs
    ):
        super(TrainerIsIn, self).__init__(**kwargs)
        self.is_in_question = is_in_question
        if is_in_question is None:
            self.is_in_question = self.question_model()

    def question_model(self):
        input = keras.Input(shape=(2 * self.wire_dimension))
        output = keras.layers.Dense(self.wire_dimension, activation=tf.nn.relu)(input)
        output = keras.layers.Dense(self.wire_dimension / 2, activation=tf.nn.relu)(output)
        output = keras.layers.Dense(1)(output)
        return keras.Model(inputs=input, outputs=output)

    # @tf.function(jit_compile=True)
    def compute_loss(self, outputs, tests):
        location, answer_prob = self._get_answer_prob(outputs, tests)
        labels = tf.one_hot(location, answer_prob.shape[1])
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=answer_prob, labels=labels)
        return loss

    # @tf.function(jit_compile=True)
    def _get_answer_prob(self, outputs, tests):
        num_wires = outputs.shape[1] // self.wire_dimension
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
        return location,answer_prob

    def get_probabilities(self, diagrams, tests):
        inputs, params = self.batch_diagrams([diagrams])
        outputs = self.call((inputs, params))
        answer_prob = self._get_answer_prob(outputs, tests)
        return answer_prob

    def get_config(self):
        config = super().get_config()
        config.update({
            "is_in_question": self.is_in_question
        })
        return config

    def get_accuracy(discocirc_trainer, dataset):
        diagrams = [data[0] for data in dataset]
        discocirc_trainer.diagrams = diagrams
        discocirc_trainer.get_parameters_from_diagrams(diagrams)
        location_predicted = []
        location_true = []
        for i in range(len(dataset)):
            print('predicting {} / {}'.format(i, len(dataset)), end='\r')
            probs = discocirc_trainer.get_probabilities(dataset[i][0],
                                                        dataset[i][1])
            location_predicted.append(np.argmax(probs))
            location_true.append(dataset[i][1][1])
        accuracy = accuracy_score(location_true, location_predicted)
        return accuracy