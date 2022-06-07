import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import pickle
import tensorflow as tf
from tensorflow import keras

from network.utils import get_fast_nn_functor, initialize_boxes


class DisCoCircTrainer(keras.Model):
    def __init__(self, nn_boxes, wire_dimension, is_in_question=None, compiled_dataset=None, **kwargs):
        super(DisCoCircTrainer, self).__init__(**kwargs)
        self.nn_boxes = nn_boxes
        self.trainable_models = [box.model for box in nn_boxes.values()]
        self.wire_dimension = wire_dimension
        self.nn_functor = get_fast_nn_functor(self.nn_boxes, wire_dimension)
        self.dataset = compiled_dataset
        if is_in_question is None:
            self.is_in_question = self.question_model()
        else:
            self.is_in_question = is_in_question
        self.loss_tracker = keras.metrics.Mean(name="loss")
        
    @staticmethod
    def from_lexicon(lexicon, wire_dimension, **kwargs):
        nn_boxes, trainable_models = initialize_boxes(lexicon, wire_dimension)
        return DisCoCircTrainer(nn_boxes, wire_dimension, compiled_dataset=None, **kwargs)

    def save_models(self, path):
        with open(path, "wb") as f:
            pickle.dump((self.nn_boxes, self.wire_dimension, self.is_in_question), f)
    
    @staticmethod
    def load_models(path):
        with open(path, "rb") as f:
            nn_boxes, wire_dimension, is_in_question = pickle.load(f)
        return DisCoCircTrainer(nn_boxes, wire_dimension, is_in_question, compiled_dataset=None)

    def compile_dataset(self, dataset, validation = False):
        model_dataset = []
        count = 0
        for context_circuit, test in dataset:
            print(count + 1, "/", len(dataset), end="\r")
            count += 1
            context_circuit_model = self.nn_functor(context_circuit)
            model_dataset.append([context_circuit_model.model, test])
        if validation:
            self.validation_dataset = model_dataset
        else:
            self.dataset = model_dataset
            self.dataset_size = len(dataset)

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
            loss, grd = self.train_step_for_sample(self.dataset[int(idx.numpy())])
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
            loss = self.compute_loss(context_circuit_model, test)
            grad = tape.gradient(loss, self.trainable_weights, unconnected_gradients=tf.UnconnectedGradients.ZERO)
        return loss, grad
    
    @tf.function
    def compute_loss(self, context_circuit_model, test):
        person, location = test
        answer_prob = self.call((context_circuit_model, person))
        labels = tf.one_hot(location, len(answer_prob))
        return tf.nn.softmax_cross_entropy_with_logits(logits=answer_prob, labels=labels)

    @tf.function
    def call(self, circ_person):
        circ, person = circ_person
        output_vector = circ(tf.convert_to_tensor([[]]))[0]
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
        answer_prob = tf.nn.softmax(answer_prob)
        return answer_prob

    def fit(self, epochs, batch_size=32, **kwargs):
        if self.dataset is None:
            raise ValueError("Dataset not compiled")
        input_index_dataset = tf.data.Dataset.range(self.dataset_size)
        input_index_dataset = input_index_dataset.shuffle(self.dataset_size)
        input_index_dataset = input_index_dataset.batch(batch_size)
        return super(DisCoCircTrainer, self).fit(input_index_dataset, epochs=epochs, **kwargs)
