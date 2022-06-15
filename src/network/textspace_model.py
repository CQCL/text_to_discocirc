import pickle
import tensorflow as tf
from tensorflow import keras

from network.circuit_to_textspace import TextSpace
from network.trainer_base_class import DisCoCircTrainerBase
from network.utils import get_classification_vocab

class DisCoCircTrainerTextspace(DisCoCircTrainerBase):
    def __init__(self, 
                 nn_boxes,
                 wire_dimension,
                 max_wire_num = 20,
                 textspace_dimension = 200,
                 classification_vocab = None,
                 **kwargs):
        super().__init__(nn_boxes, wire_dimension, **kwargs)
        self.circuit_to_textspace = TextSpace(max_wire_num, textspace_dimension)
        self.max_wire_num = max_wire_num
        self.textspace_dimension = textspace_dimension
        self.classification_vocab = classification_vocab
        if self.classification_vocab is None:
            if self.lexicon is None:
                raise ValueError("Either lexicon or classification_vocab must be provided")
            self.classification_vocab = get_classification_vocab(self.lexicon)
        self.qna_classifier = self.qna_classifier()

    def save_models(self, path):
        kwargs = {
            "nn_boxes": self.nn_boxes,
            "wire_dimension": self.wire_dimension,
            "max_wire_num": self.max_wire_num,
            "textspace_dimension": self.textspace_dimension
        }
        with open(path, "wb") as f:
            pickle.dump(kwargs, f)

    def qna_classifier(self):
        input = keras.Input(shape=(2 * self.textspace_dimension))
        output = keras.layers.Dense(self.textspace_dimension, activation=tf.nn.relu)(input)
        output = keras.layers.Dense(self.textspace_dimension / 2, activation=tf.nn.relu)(input)
        output = keras.layers.Dense(self.textspace_dimension / 4, activation=tf.nn.relu)(input)
        output = keras.layers.Dense(len(self.classification_vocab), activation=tf.nn.softmax)(output)
        return keras.Model(inputs=input, outputs=output)
    
    @tf.function
    def compute_loss(self, context_circuit_model, test):
        question_circuit_model, answer_word = test
        answer_prob = self.call((context_circuit_model, question_circuit_model))
        answer_index = self.classification_vocab.index(answer_word)
        true_answer = tf.one_hot(answer_index, answer_prob.shape[0])
        return keras.metrics.mean_squared_error(true_answer, answer_prob)

    @tf.function
    def call(self, context_question):
        context_circuit, question_circuit = context_question
        context_vector = self.circuit_to_textspace(
            context_circuit(tf.convert_to_tensor([[]]))[0]
        )
        question_vector = self.circuit_to_textspace(
            question_circuit(tf.convert_to_tensor([[]]))[0]
        )
        classifier_input = tf.concat([context_vector, question_vector], axis=1)
        return self.qna_classifier(classifier_input)

