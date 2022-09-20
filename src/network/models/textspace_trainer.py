import pickle

import numpy as np
import tensorflow as tf
from tensorflow import keras

from network.models.trainer_base_class import DisCoCircTrainerBase
from network.utils.circuit_to_textspace import TextSpace
from network.utils.utils import get_classification_vocab


class DisCoCircTrainerTextspace(DisCoCircTrainerBase):
    def __init__(self, 
                 nn_boxes,
                 wire_dimension,
                 max_wire_num = 20,
                 textspace_dimension = 200,
                 latent_dimension = None,
                 classification_vocab = None,
                 qna_classifier_model = None,
                 space_expansion = None,
                 space_contraction = None,
                 **kwargs):
        super().__init__(nn_boxes, wire_dimension, **kwargs)
        self.circuit_to_textspace = TextSpace(
            wire_dimension, 
            max_wire_num, 
            textspace_dimension, 
            latent_dimension, 
            space_expansion,
            space_contraction
        )            
        self.max_wire_num = max_wire_num
        self.textspace_dimension = textspace_dimension
        self.classification_vocab = classification_vocab
        # if classification_vocab is empty, construct it from lexicon
        if self.classification_vocab is None:
            if self.lexicon is None:
                raise ValueError("Either lexicon or classification_vocab must be provided")
            self.classification_vocab = get_classification_vocab(self.lexicon)
        self.qna_classifier_model = qna_classifier_model
        if self.qna_classifier_model is None:    
            self.qna_classifier_model = self.qna_classifier()

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
            "qna_classifier_model": self.qna_classifier_model,
            "space_expansion": self.circuit_to_textspace.space_expansion,
            "space_contraction": self.circuit_to_textspace.space_contraction,
            "wire_dimension": self.wire_dimension,
            "max_wire_num": self.max_wire_num,
            "textspace_dimension": self.textspace_dimension,
            "classification_vocab": self.classification_vocab,
            "lexicon": self.lexicon
        }
        with open(path, "wb") as f:
            pickle.dump(kwargs, f)

    def qna_classifier(self):
        input = keras.Input(shape=(2 * self.textspace_dimension))
        # output = keras.layers.Dense(self.textspace_dimension, activation=tf.nn.relu)(input)
        output = keras.layers.Dense(self.textspace_dimension / 2, activation=tf.nn.relu)(input)
        # output = keras.layers.Dense(self.textspace_dimension / 4, activation=tf.nn.relu)(input)
        output = keras.layers.Dense(len(self.classification_vocab), activation=tf.nn.softmax)(output)
        return keras.Model(inputs=input, outputs=output)

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
        true_answer = tf.one_hot(answer_index, answer_prob.shape[1])
        return keras.metrics.mean_squared_error(true_answer, answer_prob)

    @tf.function
    def call(self, context_question):
        """
        The model's forward pass
        """
        context_circuit, question_circuit = context_question
        context_vector = self.circuit_to_textspace(
            context_circuit(tf.convert_to_tensor([[]]))
        )
        question_vector = self.circuit_to_textspace(
            question_circuit(tf.convert_to_tensor([[]]))
        )
        classifier_input = tf.concat([context_vector, question_vector], axis=1)
        return self.qna_classifier_model(classifier_input)
