import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from tensorflow import keras

from network.big_network_models.one_network_trainer_base import OneNetworkTrainerBase
from network.utils.circuit_to_textspace import TextSpace
from network.utils.utils import create_feedforward_network


class TextspaceOneNetworkTrainer(OneNetworkTrainerBase):
    def __init__(self,
                 wire_dimension,
                 max_wire_num=20,
                 lexicon=None,
                 textspace_dimension=None,
                 latent_dimension=None,
                 expansion_hidden_layers=None,
                 contraction_hidden_layers=None,
                 vocab_dict=None,
                 qna_classifier_model=None,
                 circuit_to_textspace=None,
                 qna_hidden_layers=None,
                 **kwargs
            ):
        super(TextspaceOneNetworkTrainer, self).__init__(lexicon=lexicon, wire_dimension=wire_dimension, **kwargs)

        if circuit_to_textspace is None:
            self.circuit_to_textspace = TextSpace(
                wire_dim=wire_dimension,
                textspace_dim=textspace_dimension,
                latent_dim=latent_dimension,
                expansion_hidden_layers=expansion_hidden_layers,
                contraction_hidden_layers=contraction_hidden_layers,
            )
        else:
            self.circuit_to_textspace = circuit_to_textspace

        self.max_wire_num = max_wire_num
        self.textspace_dimension = textspace_dimension

        if vocab_dict is None:
            self.vocab_dict = {}
            for i, v in enumerate(lexicon):
                self.vocab_dict[v.name] = i
        else:
            self.vocab_dict = vocab_dict

        if qna_classifier_model is None:
            self.qna_classifier_model = create_feedforward_network(
                input_dim = 2 * self.textspace_dimension,
                output_dim = len(self.vocab_dict),
                hidden_layers = qna_hidden_layers
            )
        else:
            self.qna_classifier_model = qna_classifier_model


    # @tf.function(jit_compile=True)
    def compute_loss(self, outputs, tests):
        location, answer_prob = self._get_answer_prob(outputs, tests)
        return tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=answer_prob,
                labels=[self.vocab_dict[location[i]] for i in range(len(location))]
        )

    # @tf.function(jit_compile=True)
    def _get_answer_prob(self, outputs, question_circuit):
        context_vector = self.circuit_to_textspace(
            outputs
        )

        question_vector = self.circuit_to_textspace(
            question_circuit(tf.convert_to_tensor([[]]))
        )

        classifier_input = tf.concat([context_vector, question_vector], axis=1)
        return self.qna_classifier_model(classifier_input)

    def get_accuracy(self, dataset):
        diagrams = [data[0] for data in dataset]
        questions = [data[1][0] for data in dataset]
        # self.diagrams = diagrams

        diagram_parameters = self.get_parameters_from_diagrams(diagrams)
        self.pad_parameters(diagram_parameters)
        self.get_block_diag_paddings(diagram_parameters)

        question_diagram_parameters = self.get_parameters_from_diagrams(questions)
        self.pad_parameters(question_diagram_parameters)
        self.get_block_diag_paddings(question_diagram_parameters)

        location_predicted = []
        location_true = []
        for i in range(len(dataset)):
            print('predicting {} / {}'.format(i, len(dataset)), end='\r')
            diagrams_params = [diagram_parameters[repr(dataset[i][0])]]
            batched_params = self.batch_diagrams(diagrams_params)
            outputs = self.call(batched_params)
            _, answer_prob = self._get_answer_prob(outputs, [question_diagram_parameters[repr(dataset[i][1][0])]])

            location_predicted.append(self.get_prediction_result(answer_prob))
            location_true.append(self.get_expected_result(dataset[i][1][1]))

        accuracy = accuracy_score(location_true, location_predicted)
        return accuracy

    def get_config(self):
        config = super().get_config()
        config.update({
            "qna_classifier_model": self.qna_classifier_model,
            "space_expansion": self.circuit_to_textspace.space_expansion,
            "space_contraction": self.circuit_to_textspace.space_contraction,
            "wire_dimension": self.wire_dimension,
            "max_wire_num": self.max_wire_num,
            "textspace_dimension": self.textspace_dimension,
            "classification_vocab": self.classification_vocab,
            "vocab_dict": self.vocab_dict
        })
        return config

    def get_prediction_result(self, call_result):
        return np.argmax(call_result)

    def get_expected_result(self, given_value):
        return self.vocab_dict[given_value]
