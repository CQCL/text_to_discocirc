import tensorflow as tf
from tensorflow import keras


class MyDenseLayer(keras.layers.Layer):
    def call(self, x, kernel, bias):
        return tf.einsum("bi,bij->bj", x, kernel) + bias


class NeuralDisCoCirc(keras.Model):
    def __init__(self, wire_dimension, lexicon):
        super().__init__()
        self.wire_dimension = wire_dimension
        self.dense_layer = MyDenseLayer()
        self.initialize_lexicon_weights(lexicon)
    
    @tf.function(jit_compile=True)
    def call(self, x, weights, biases):
        weights = tf.transpose(weights, perm=[1,0,2,3])
        biases = tf.transpose(biases, perm=[1,0,2])
        output = x
        for i in range(tf.shape(weights)[0]):
            output = self.dense_layer(output, weights[i], biases[i])
        return output
    
    def initialize_lexicon_weights(self, lexicon):
        self.lexicon_weights = {}
        self.lexicon_biases = {}
        self.states = {}
        for word in lexicon:
            input_dim = len(word.dom) * self.wire_dimension
            output_dim = len(word.cod) * self.wire_dimension
            if input_dim == 0:
                self.states[word] = self.add_weight(
                    shape = (output_dim,), 
                    initializer = "glorot_uniform",
                    trainable = True
                )
            else:
                self.lexicon_weights[word] = self.add_weight(
                    shape = (input_dim, output_dim), 
                    initializer = "glorot_uniform",
                    trainable = True
                )
                self.lexicon_biases[word] = self.add_weight(
                    shape = (output_dim,), 
                    initializer = "glorot_uniform",
                    trainable = True
                )


import pickle
with open('data/task_vocab_dicts/en_qa1_train.p', 'rb') as f:
    vocab = pickle.load(f)

neural_discocirc = NeuralDisCoCirc(20, vocab)

total_dim = 500
depth = 100
batch = 32
x = tf.random.normal(shape=(batch, total_dim))
weights = tf.random.normal(shape=(batch, depth, total_dim, total_dim))
biases = tf.random.normal(shape=(batch, depth, total_dim))

neural_discocirc(x, weights, biases)

from time import time
start_time = time()
for i in range(1000):
    neural_discocirc(x, weights, biases)
end_time = time()
print('total time', end_time - start_time, 'seconds')
