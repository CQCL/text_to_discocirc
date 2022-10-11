import pickle

import numpy as np
import tensorflow as tf
from tensorflow import keras

from network.individual_networks_models.individual_networks_trainer_base_class import \
    IndividualNetworksTrainerBase
from network.utils.utils import get_classification_vocab


class LSTMIndividualNetworksTrainer(IndividualNetworksTrainerBase):
    def __init__(self, 
                 nn_boxes,
                 wire_dimension,
                 lstm_dimension = 10,
                 classification_vocab = None,
                 lstm_model = None,
                 classifier_model = None,
                 **kwargs):
        super().__init__(nn_boxes, wire_dimension, **kwargs)
        self.lstm_dimension = lstm_dimension
        self.classification_vocab = classification_vocab
        self.lstm_model = lstm_model
        self.classifier_model = classifier_model
        # if classification_vocab is empty, construct it from lexicon
        if self.classification_vocab is None:
            if self.lexicon is None:
                raise ValueError("Either lexicon or classification_vocab must be provided")
            self.classification_vocab = get_classification_vocab(self.lexicon)
        if self.lstm_model is None:
            self.lstm_model = keras.layers.LSTM(lstm_dimension)
        if self.classifier_model is None:
            self.classifier_model = keras.layers.Dense(len(self.classification_vocab), activation="softmax")


    def compile_dataset(self, dataset, validation = False):
        """
        applies the nn_functor to the list of context and question circuit diagrams,
        and saves these
        """
        model_dataset = []
        count = 0
        for context_circuit, test in dataset:
            print(count + 1, "/", len(dataset), end="\r")
            count += 1
            context_circuit_model = self.nn_functor(context_circuit)
            question_circuit_model = self.nn_functor(test[0])
            model_dataset.append([context_circuit_model.model, (question_circuit_model.model, test[1])])
        if validation:
            self.validation_dataset = model_dataset
        else:
            self.dataset = model_dataset
            self.dataset_size = len(dataset)

    def save_models(self, path):
        kwargs = {
            "nn_boxes": self.nn_boxes,
            "lstm_dimension": self.lstm_dimension,
            "wire_dimension": self.wire_dimension,
            "classification_vocab": self.classification_vocab,
            "lexicon": self.lexicon,
            "lstm_model": self.lstm_model,
            "classifier_model": self.classifier_model
        }
        with open(path, "wb") as f:
            pickle.dump(kwargs, f)

    def get_prediction_result(self, model_output):
        return self.classification_vocab[np.argmax(model_output)]

    def get_expected_result(self, given_value):
        return given_value
    
    @tf.function
    def compute_loss(self, context_circuit_model, test):
        # test is a tuple containing (question_circuit_model, answer_word)
        question_circuit_model, answer_word = test
        answer_prob = self.call((context_circuit_model, question_circuit_model))
        answer_index = self.classification_vocab.index(answer_word)
        return tf.nn.sparse_softmax_cross_entropy_with_logits(logits=answer_prob,
                                                          labels=answer_index)

    @tf.function
    def call(self, context_question):
        """
        The model's forward pass
        """
        context_circuit, question_circuit = context_question
        context_vector = context_circuit(tf.convert_to_tensor([[]]))
        question_vector = question_circuit(tf.convert_to_tensor([[]]))
        num_wires_context = context_vector.shape[1] // self.wire_dimension
        output_wires_context = tf.split(context_vector, num_wires_context, axis=1)
        num_wires_question = question_vector.shape[1] // self.wire_dimension
        output_wires_question = tf.split(question_vector, num_wires_question, axis=1)
        outputs = output_wires_context + output_wires_question
        outputs = tf.stack(outputs, axis=1)
        outputs = self.lstm_model(outputs)
        outputs = self.classifier_model(outputs)
        return outputs
