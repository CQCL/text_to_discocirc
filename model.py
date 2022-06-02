import os

os.environ["CUDA_VISIBLE_DEVICES"]="1"

import tensorflow as tf
from tensorflow import keras

from utils import get_nn_functor, initialize_boxes

class DisCoCircTrainer(keras.Model):
    def __init__(self, lexicon, wire_dimension, **kwargs):
        super(DisCoCircTrainer, self).__init__(**kwargs)
        self.lexicon = lexicon
        self.wire_dimension = wire_dimension
        self.nn_boxes, self.trainable_models = initialize_boxes(lexicon, wire_dimension)
        self.nn_functor = get_nn_functor(self.nn_boxes, wire_dimension)
        self.loss_tracker = keras.metrics.Mean(name="loss")

    def compile_dataset(self, dataset):
        self.dataset_size = len(dataset)
        self.dataset = []
        for context_circuit, test in dataset:
            context_circuit_model = self.nn_functor(context_circuit)
            self.dataset.append([context_circuit_model.model, test])
    
    def train_step(self, batch):
        losses = 0
        grads = None
        for idx in batch:
            loss, grd = self.train_step_for_sample(self.dataset[idx])
            losses += loss
            if grads is None:
                grads = grd
            else:
                grads = [g1 + g2 for g1, g2 in zip(grads, grd)]

        self.optimizer.apply_gradients((grad, weights)
            for (grad, weights) in zip(grads, self.trainable_weights)
            if grad is not None)

        self.loss_tracker.update_state(losses)
        return {
            "loss": self.loss_tracker.result(),
        }

    # @tf.function
    def train_step_for_sample(self, dataset):
        with tf.GradientTape() as tape:
            context_circuit_model, test = dataset
            output_vector = context_circuit_model(tf.convert_to_tensor([[]]))
            loss = self.compute_loss(output_vector, test)
            grad = tape.gradient(loss, self.trainable_weights, unconnected_gradients=tf.UnconnectedGradients.ZERO)
        return loss, grad
    
    #TODO implement loss function
    @tf.function
    def compute_loss(self, output_vector, test):
        # person, location = test
        return tf.reduce_sum(1 - output_vector)

