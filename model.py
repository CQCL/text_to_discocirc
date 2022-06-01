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

