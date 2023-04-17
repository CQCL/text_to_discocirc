from discopy.rigid import Id

from discocirc.diag.frame import Frame, Functor


def remove_the(digram):
    def f_box(box):
        if box.name.lower() == "the" and box.dom == box.cod:
            return Id(box.dom)
        return box
    f = Functor(ob=lambda x: x, ar=f_box, frame=lambda x: x)
    return f(digram)

def remove_to_be(digram):
    def is_to_be(str):
        str = str.lower()
        if str =="am" or str == "is" or str == "are" or str == "was" or str == "were":
            return True
        return False
    f = Functor(ob=lambda x: x, ar=lambda x: x, frame=lambda x: frame_to_id(x, is_to_be))
    return f(digram)

def remove_relative_pronouns(digram):
    def is_relative_pronouns(str):
        str = str.lower()
        if str == "who" or str == "whom" or str == "whose" or str == "which" or str == "that":
            return True
        return False
    f = Functor(ob=lambda x: x, ar=lambda x: x, frame=lambda x: frame_to_id(x, is_relative_pronouns))
    return f(digram)

def frame_to_id(box, name_condition):
    if name_condition(box.name) and \
        len(box.insides) == 1 and \
        box.dom == box.cod == box.insides[0].dom == box.insides[0].cod:
        return box.insides[0]
    return box
