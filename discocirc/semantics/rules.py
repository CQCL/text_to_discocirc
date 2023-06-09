from discopy.rigid import Id

from discocirc.diag.frame import Functor

def remove_articles(diagram):
    def f_box(box):
        str = box.name.lower()
        if (str == "the" or str == "a" or str == "an") and \
            box.dom == box.cod and len(box.dom) == 1:
            return Id(box.dom)
        return box
    f = Functor(ob=lambda x: x, ar=f_box, frame=lambda x: x)
    return f(diagram)

def remove_to_be(diagram):
    def is_to_be(str):
        str = str.lower()
        if str =="am" or str == "is" or str == "are" or str == "was" or str == "were":
            return True
        return False
    f = Functor(ob=lambda x: x, ar=lambda x: x, frame=lambda x: frame_to_id(x, is_to_be))
    return f(diagram)

def remove_relative_pronouns(diagram):
    def is_relative_pronouns(str):
        str = str.lower()
        if str == "who" or str == "whom" or str == "whose" or str == "which" or str == "that":
            return True
        return False
    f = Functor(ob=lambda x: x, ar=lambda x: x, frame=lambda x: frame_to_id(x, is_relative_pronouns))
    return f(diagram)

def frame_to_id(box, name_condition):
    if name_condition(box.name) and \
        len(box.insides) == 1 and \
        box.dom == box.cod == box.insides[0].dom == box.insides[0].cod:
        return box.insides[0]
    return box
