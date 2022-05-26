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
        self.nn_boxes, self.trainable_models = self.initialize_boxes(vocab, wire_dimension)
        self.nn_functor = get_nn_functor(self.nn_boxes, wire_dimension)

    #TODO do not hard-code hidden layers
    def initialize_boxes(self, vocab, wire_dimension):
        nn_boxes = {}
        trainable_models=[]
        for word in vocab:
            nn_boxes[word] = Network.dense_model(
                len(word.dom) * wire_dimension,
                len(word.cod) * wire_dimension,
                [10, 10] #hidden layers
            )
            trainable_models.append(nn_boxes[word].model)
        return nn_boxes, trainable_models

    def prepare_dataset(self, dataset):
        self.dataset_size = len(dataset)
        self.dataset = []
        for context_circuit, test in dataset:
            self.dataset.append([self.nn_functor(context_circuit), test])
    
    def train_step(self, batch):
        with tf.GradientTape() as tape:
            loss = 0
            for context_circuit_model, test in batch:
                output_vector = context_circuit_model()
                loss += self.compute_loss(output_vector, test)
        grads = tape.gradient(loss, self.trainable_weights)

        self.optimizer.apply_gradients((grad, weights)
            for (grad, weights) in zip(grads, self.trainable_weights)
            if grad is not None)

        self.loss_tracker.update_state(loss)
        return {
            "loss": self.loss_tracker.result(),
        }
    
    #TODO implement loss function
    def compute_loss(self, output_vector, test):
        return 0

