#%%
from discopy.rigid import Ty, Box, Diagram, Functor, Id


def remove_the(digram):
    def star_removal_ar(box):
        if box.name.lower() == "the" and box.dom == box.cod:
            return Id(box.dom)
        return box
    f = Functor(ob=lambda x: x, ar=star_removal_ar)
    return f(digram)
