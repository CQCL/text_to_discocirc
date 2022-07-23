from abc import ABC, abstractmethod
import pickle
import tensorflow as tf
from tensorflow import keras

from network.utils import get_fast_nn_functor, initialize_boxes


class DisCoCircTrainerBase(ABC, keras.Model):
    def __init__(self, nn_boxes, wire_dimension, compiled_dataset=None, lexicon=None, **kwargs):
        super(DisCoCircTrainerBase, self).__init__(**kwargs)
        self.nn_boxes = nn_boxes
        self.trainable_models = [box.model for box in nn_boxes.values()]
        self.wire_dimension = wire_dimension
        self.nn_functor = get_fast_nn_functor(self.nn_boxes, wire_dimension)
        self.dataset = compiled_dataset
        self.lexicon = lexicon
        self.loss_tracker = keras.metrics.Mean(name="loss")    

    @classmethod
    def from_lexicon(cls, lexicon, wire_dimension, hidden_layers=[10, 10], **kwargs):
        """
        Factory method to create a DisCoCircTrainer from a lexicon.

        Parameters
        ----------
        lexicon : list
            list of discopy boxes in the lexicon.
        wire_dimension : int
            dimension of the noun wires.
        """
        nn_boxes, trainable_models = initialize_boxes(lexicon, wire_dimension, hidden_layers)
        return cls(nn_boxes, wire_dimension, compiled_dataset=None, lexicon=lexicon, **kwargs)

    def save_models(self, path):
        kwargs = {
            "nn_boxes": self.nn_boxes,
            "wire_dimension": self.wire_dimension,
        }
        with open(path, "wb") as f:
            pickle.dump(kwargs, f)
    
    @classmethod
    def load_models(cls, path):
        with open(path, "rb") as f:
            kwargs = pickle.load(f)
        return cls(**kwargs)

    def compile_dataset(self, dataset, validation = False):
        """
        applies the nn_functor to the list of context circuit diagrams,
        and saves these
        """
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

    @abstractmethod
    def compute_loss(self, context_circuit_model, test):
        pass

    def fit(self, epochs, batch_size=32, **kwargs):
        if self.dataset is None:
            raise ValueError("Dataset not compiled")
        input_index_dataset = tf.data.Dataset.range(self.dataset_size)
        input_index_dataset = input_index_dataset.shuffle(self.dataset_size)
        input_index_dataset = input_index_dataset.batch(batch_size)
        return super(DisCoCircTrainerBase, self).fit(input_index_dataset, epochs=epochs, **kwargs)
