import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from tensorflow import keras

from network.big_network_models.one_network_trainer_base import OneNetworkTrainerBase
from network.utils.utils import create_feedforward_network


class AddLogitsOneNetworkTrainer(OneNetworkTrainerBase):
    def __init__(self,
                 softmax_logits=True,
                 vocab_dict=None,
                 lexicon=None,
                 hidden_layers=[10, 10],
                 wire_dimension=10,
                 is_in_question=None,
                 is_in_hidden_layers=[10, 10],
                 **kwargs
            ):
        super(AddLogitsOneNetworkTrainer, self).__init__(lexicon=lexicon, wire_dimension=wire_dimension, hidden_layers=hidden_layers, **kwargs)

        self.softmax_logits = softmax_logits

        if vocab_dict is None:
            self.vocab_dict = {}
            for i, v in enumerate(lexicon):
                self.vocab_dict[v.name] = i
        else:
            self.vocab_dict = vocab_dict

        if is_in_question is None:
            self.is_in_question = create_feedforward_network(
                input_dim = 2 * wire_dimension,
                output_dim = len(self.vocab_dict),
                hidden_layers = is_in_hidden_layers
            )
        else:
            self.is_in_question = is_in_question

    # @tf.function(jit_compile=True)
    def compute_loss(self, outputs, tests):
        location, answer_prob = self._get_answer_prob(outputs, tests)
        return tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=answer_prob,
                labels=[self.vocab_dict[location[i]] for i in range(len(location))]
        )

    # @tf.function(jit_compile=True)
    def _get_answer_prob(self, outputs, tests):
        num_wires = outputs.shape[1] // self.wire_dimension
        output_wires = tf.split(outputs, num_wires, axis=1)
        tests = np.array(tests).T
        person, location = tests[0], tests[1]
        person = [(int(person), i) for i, person in enumerate(person)]
        person_vectors = tf.gather_nd(output_wires, person)

        logit_sum = [tf.zeros(len(self.vocab_dict)) for _ in range(len(location))]
        for i in range(num_wires):
            location_vectors = output_wires[i]

            answer = self.is_in_question(
                    tf.concat([person_vectors, location_vectors], axis=1)
            )

            logit = tf.convert_to_tensor(answer)
            if self.softmax_logits:
                logit = tf.nn.softmax(logit)

            logit_sum = tf.math.add(logit, logit_sum)

        return location, logit_sum

    def get_config(self):
        config = super().get_config()
        config.update({
            "wire_dimension": self.wire_dimension,
            "is_in_question": self.is_in_question,
            "vocab_dict": self.vocab_dict,
        })
        return config

    def get_prediction_result(self, call_result):
        return np.argmax(call_result)

    def get_expected_result(self, given_value):
        return self.vocab_dict[given_value]
