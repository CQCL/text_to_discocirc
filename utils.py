from discopy.monoidal import Functor
from discopy import PRO
from discopy.rigid import Ty, Box
from Network import Network

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
        nn_boxes[word] = Network.dense_model(
            len(word.dom) * wire_dimension,
            len(word.cod) * wire_dimension,
            hidden_layer_dims=hidden_layers
        )
        trainable_models.append(nn_boxes[word].model)
    return nn_boxes, trainable_models


def get_star_removal_functor():
    def star_removal_ob(ty):
        return Ty() if ty.name == "*" else ty

    def star_removal_ar(box):
        return Box(box.name, f(box.dom), f(box.cod))

    f = Functor(ob=star_removal_ob, ar=star_removal_ar)
    return f
    