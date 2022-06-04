from discopy.monoidal import Functor
from discopy import PRO

from network.network import Network
from tensorflow import keras


def get_nn_functor(nn_boxes, wire_dim):
    def neural_ob(t):
        return PRO(len(t) * wire_dim)
    def neural_ar(box):
        return nn_boxes[box]
    f = Functor(ob=neural_ob, ar=neural_ar, ar_factory=Network)
    return f

#TODO do not hard-code hidden layers
def initialize_boxes(lexicon, wire_dimension, hidden_layers=[10, 10]):
    nn_boxes = {}
    trainable_models=[]
    for word in lexicon:
        #TODO add names to the model. It does not like [box] and [\box].
        name = word.name
        if '\\' in name:
            name = name.replace('\\', '')
            name = name[1:-1] + '_end'
        elif '[' in name:
            name = name[1:-1] + '_begin'
        nn_boxes[word] = Network.dense_model(
            len(word.dom) * wire_dimension,
            len(word.cod) * wire_dimension,
            hidden_layer_dims=hidden_layers,
            name = name
        )
        trainable_models.append(nn_boxes[word].model)
    return nn_boxes, trainable_models

