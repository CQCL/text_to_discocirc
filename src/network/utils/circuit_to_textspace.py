import tensorflow as tf
from tensorflow import keras

class TextSpace(keras.Model):
    """
    The model that maps the output vector of a circuit to a 'textspace' vector.
    This 'textspace' vector has dimension 'self.textspace_dim'
    """
    def __init__(self, wire_dim, max_wire_num, textspace_dim, latent_dim=None, space_expansion=None, space_contraction=None):
        super(TextSpace, self).__init__()
        self.wire_dim = wire_dim
        self.max_wire_num = max_wire_num
        self.textspace_dim = textspace_dim

        if latent_dim is None:
            self.latent_dim = max_wire_num * wire_dim
        else:
            self.latent_dim = latent_dim

        if space_expansion is None:
            self.space_expansion = self.define_model(self.wire_dim, self.latent_dim)
        else:
            self.space_expansion = space_expansion

        if space_contraction is None:
            self.space_contraction = self.define_model(self.latent_dim, self.textspace_dim)
        else:
            self.space_contraction = space_contraction

    def call(self, circuit_vector):
        circuit_vector = circuit_vector[0]
        num_wires = circuit_vector.shape[0] // self.wire_dim
        wire_vectors = tf.split(circuit_vector, num_or_size_splits=num_wires, axis=0)
        wire_vectors = tf.convert_to_tensor(wire_vectors)
        wire_vectors = self.space_expansion(wire_vectors)
        latent_vector = tf.reduce_sum(wire_vectors, axis=0)
        textspace_vector = self.space_contraction(tf.expand_dims(latent_vector, axis=0))
        return textspace_vector

    def define_model(self, in_dim, out_dim):
        """
        Generates the space_contraction and space_expansion models
        """
        factor = round((out_dim/in_dim)**(1./3))
        input = keras.Input(shape=(in_dim))
        output = keras.layers.Dense(in_dim * factor, activation=tf.nn.relu)(input)
        # output = keras.layers.Dense(in_dim * factor * factor, activation=tf.nn.relu)(output)
        output = keras.layers.Dense(out_dim)(output)
        return keras.Model(inputs=input, outputs=output)
