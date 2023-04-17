#%%
from discopy.rigid import Ty, Box, Diagram, Functor, Id

from discocirc.diag.frame import Frame


def remove_the(digram):
    def f_box(box):
        if box.name.lower() == "the" and box.dom == box.cod:
            return Id(box.dom)
        return box
    f = Functor(ob=lambda x: x, ar=f_box)
    return f(digram)

def remove_is(digram):
    def f_box(box):
        if isinstance(box, Frame) and \
            box.name.lower() == "is" and \
            len(box.insides) == 1 and \
            box.dom == box.cod == box.insides[0].dom == box.insides[0].cod:
            return box.insides[0]
        return box
    f = Functor(ob=lambda x: x, ar=f_box)
    return f(digram)
