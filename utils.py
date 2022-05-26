from discopy.neural import Network
from discopy.monoidal import Functor
from discopy import PRO
from discopy.monoidal import Ty, Box

def get_nn_functor(nn_boxes, wire_dim):
    def neural_ob(t):
        return PRO(len(t) * wire_dim)
    def neural_ar(box):
        return nn_boxes[box]
    f = Functor(ob=neural_ob, ar=neural_ar, ar_factory=Network)
    return f

def get_star_removal_functor():
    def star_removal_ob(ty):
        return Ty() if ty.name == "*" else ty

    def star_removal_ar(box):
        return Box(box.name, f(box.dom), f(box.cod))

    f = Functor(ob=star_removal_ob, ar=star_removal_ar)
    return f