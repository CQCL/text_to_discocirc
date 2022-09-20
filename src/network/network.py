from discopy import monoidal
from discopy.monoidal import PRO
import tensorflow as tf
from tensorflow import keras


class Network(monoidal.Box):
    """ Implements tensorflow neural networks
    >>> a = Network.dense_model(16, 12, [5, 6])
    >>> b = Network.dense_model(12, 2, [5])
    >>> assert (a >> b).model.layers[1:] == a.model.layers[1:] + b.model.layers[1:]
    >>> assert (a >> Network.id(12)).model == a.model
    """

    def __init__(self, dom, cod, model):
        self.model = model
        super().__init__("Network", dom, cod)

    def then(self, other):
        inputs = keras.Input(shape=(len(self.dom),))
        output = self.model(inputs)
        output = other.model(output)
        composition = keras.Model(inputs=inputs, outputs=output)
        return Network(self.dom, other.cod, composition)

    def tensor(self, other):
        dom = len(self.dom) + len(other.dom)
        cod = len(self.cod) + len(other.cod)
        inputs = keras.Input(shape=(dom,))
        model1_input_len = len(self.dom)
        model1 = keras.layers.Lambda(
            lambda x: x[:, :model1_input_len],)(inputs)
        model2 = keras.layers.Lambda(
            lambda x: x[:, model1_input_len:],)(inputs)
        model1 = self.model(model1)
        model2 = other.model(model2)
        outputs = keras.layers.Concatenate()([model1, model2])
        model = keras.Model(inputs=inputs, outputs=outputs)
        return Network(PRO(dom), PRO(cod), model)

    def swap(left, right, ar_factory=None, swap_factory=None):
        left_len = len(left)
        dim = left_len + len(right)

        inputs = keras.Input(shape=(dim,))
        model1 = keras.layers.Lambda(
            lambda x: x[:, :left_len], )(inputs)
        model2 = keras.layers.Lambda(
            lambda x: x[:, left_len:], )(inputs)

        outputs = keras.layers.Concatenate()([model2, model1])
        model = keras.Model(inputs=inputs, outputs=outputs)
        return Network(PRO(dim), PRO(dim), model)


    @staticmethod
    def id(dim):
        inputs = keras.Input(shape=(len(dim),))
        return Network(dim, dim, keras.Model(inputs=inputs, outputs=inputs))

    @staticmethod
    def dense_model(dom, cod, name=None, hidden_layer_dims=[], activation=tf.nn.relu):
        """
        Parameters
        ----------
        dom : int
            dimension of the domain
        cod : int
            dimension of the codomain
        name : str
            name of the box
        hidden_layer_dims : list of int
            list of dimensions of the hidden layers
        activation : function
            activation function
        """
        inputs = keras.Input(shape=(dom,))
        model = inputs
        for dim in hidden_layer_dims:
            model = keras.layers.Dense(dim, activation=activation, bias_initializer="glorot_uniform")(model)
        outputs = keras.layers.Dense(cod, activation=activation, bias_initializer="glorot_uniform")(model)
        model = keras.Model(inputs=inputs, outputs=outputs, name=name)
        return Network(PRO(dom), PRO(cod), model)
