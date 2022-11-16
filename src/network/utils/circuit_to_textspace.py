import tensorflow as tf
from tensorflow import keras

from network.utils.utils import create_feedforward_network


class TextSpace(keras.Model):
    """
    The model that maps the output vector of a circuit to a 'textspace' vector.
    This 'textspace' vector has dimension 'self.textspace_dim'
    """
    def __init__(self,
                 wire_dim,
                 textspace_dim,
                 latent_dim=None,
                 expansion_hidden_layers=None,
                 contraction_hidden_layers=None
            ):

        super(TextSpace, self).__init__()
        self.wire_dim = wire_dim
        self.expansion_hidden_layers = expansion_hidden_layers
        self.contraction_hidden_layers = contraction_hidden_layers

        self.space_expansion = create_feedforward_network(
            self.wire_dim,
            latent_dim,
            expansion_hidden_layers
        )

        self.space_contraction = create_feedforward_network(
            latent_dim,
            textspace_dim,
            contraction_hidden_layers
        )

    def call(self, circuit_vector):
        circuit_vector = circuit_vector[0]
        num_wires = circuit_vector.shape[0] // self.wire_dim
        wire_vectors = tf.split(circuit_vector, num_or_size_splits=num_wires, axis=0)
        wire_vectors = tf.convert_to_tensor(wire_vectors)
        expanded_vectors = self.space_expansion(wire_vectors)
        latent_vector = tf.reduce_sum(expanded_vectors, axis=0)
        textspace_vector = self.space_contraction(tf.expand_dims(latent_vector, axis=0))
        return textspace_vector
