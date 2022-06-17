from discopy.rigid import Ty
from discopy.rigid import Ty, Box
from discopy.monoidal import Functor


def init_nouns(circ):
    """
    takes in a circuit with some number of states as the initial boxes
    returns the index of the last of these initial states
    """
    for i in range(len(circ.boxes) - 1):
        if circ.boxes[i].dom == Ty() and circ.boxes[i + 1].dom != Ty():
            return i

    return -1


def get_star_removal_functor():
    def star_removal_ob(ty):
        return Ty() if ty.name == "*" else ty

    def star_removal_ar(box):
        return Box(box.name, f(box.dom), f(box.cod))

    f = Functor(ob=star_removal_ob, ar=star_removal_ar)
    return f
    
