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
        self.is_in_question = self.question_model()
        self.loss_tracker = keras.metrics.Mean(name="loss")

    def compile_dataset(self, dataset):
        self.dataset_size = len(dataset)
        self.dataset = []
        for context_circuit, test in dataset:
            context_circuit_model = self.nn_functor(context_circuit)
            self.dataset.append([context_circuit_model.model, test])

    def question_model(self):
        input = keras.Input(shape=(2 * self.wire_dimension))
        output = keras.layers.Dense(self.wire_dimension, activation=tf.nn.relu)(input)
        output = keras.layers.Dense(self.wire_dimension / 2, activation=tf.nn.relu)(output)
        output = keras.layers.Dense(1)(output)
        return keras.Model(inputs=input, outputs=output)
    
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

    @tf.function
    def train_step_for_sample(self, dataset):
        with tf.GradientTape() as tape:
            context_circuit_model, test = dataset
            output_vector = context_circuit_model(tf.convert_to_tensor([[]]))
            loss = self.compute_loss(output_vector[0], test)
            grad = tape.gradient(loss, self.trainable_weights, unconnected_gradients=tf.UnconnectedGradients.ZERO)
        return loss, grad
    
    @tf.function
    def compute_loss(self, output_vector, test):
        person, location = test
        total_wires = output_vector.shape[0] // self.wire_dimension
        person_vector = output_vector[person * self.wire_dimension : (person + 1) * self.wire_dimension]
        answer_prob = []
        for i in range(total_wires):
            location_vector = output_vector[i * self.wire_dimension : (i + 1) * self.wire_dimension]
            answer_prob.append(
                self.is_in_question(
                    tf.expand_dims(tf.concat([person_vector, location_vector], axis=0), axis=0)
                )[0][0]
            )
        answer_prob = tf.convert_to_tensor(answer_prob)
        answer_prob = tf.concat([answer_prob[:person], answer_prob[person+1:]], axis=0)
        labels = tf.one_hot(location, total_wires)
        labels = tf.concat([labels[:person], labels[person+1:]], axis=0)
        return tf.nn.softmax_cross_entropy_with_logits(logits=answer_prob, labels=labels)

