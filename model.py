import os

from utils import get_nn_functor
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import tensorflow as tf
from tensorflow import keras
from discopy.neural import Network

class DisCoCirc(keras.Model):
    def __init__(self, vocab, wire_dimension, **kwargs):
        super(DisCoCirc, self).__init__(**kwargs)
        self.vocab = vocab
        self.wire_dimension = wire_dimension
        self.nn_boxes = self.initialize_boxes(vocab, wire_dimension)
        self.nn_functor = get_nn_functor(self.nn_boxes, wire_dimension)

    #TODO do not hard-code hidden layers
    def initialize_boxes(self, vocab, wire_dimension):
        nn_boxes = {}
        for word in vocab:
            nn_boxes[word] = Network.dense_model(
                len(word.dom) * wire_dimension,
                len(word.cod) * wire_dimension,
                [10, 10] #hidden layers
            )
        return nn_boxes

