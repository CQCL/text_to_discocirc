import spacy
from discopy.monoidal import Id, Box, Swap, Ty

from discocirc.diag.frame import Functor

spacy_model = spacy.load('en_core_web_trf')


def remove_articles(diagram):
    """
    Removes articles "a", "an", "the" from a diagram.
    """
    def f_box(box):
        str = box.name.lower()
        if (str == "the" or str == "a" or str == "an") and \
            box.dom == box.cod and len(box.dom) == 1:
            return Id(box.dom)
        return box
    f = Functor(ob=lambda x: x, ar=f_box, frame=lambda x: x)
    return f(diagram)

def remove_to_be(diagram):
    """
    Removes "am", "is", "are", "was", "were" from a diagram.
    """
    def is_to_be(str):
        str = str.lower()
        if str =="am" or str == "is" or str == "are" or str == "was" or str == "were":
            return True
        return False
    f = Functor(ob=lambda x: x, ar=lambda x: x, frame=lambda x: frame_to_id(x, is_to_be))
    return f(diagram)

def remove_relative_pronouns(diagram):
    """
    Removes relative pronouns "who", "whom", "whose", "which", "that" from a diagram.
    """
    def is_relative_pronouns(str):
        str = str.lower()
        if str == "who" or str == "whom" or str == "whose" or str == "which" or str == "that":
            return True
        return False
    f = Functor(ob=lambda x: x, ar=lambda x: x, frame=lambda x: frame_to_id(x, is_relative_pronouns))
    return f(diagram)

def frame_to_id(box, name_condition):
    """
    Sets the outside frame to an identity if the name of the frame satisfies the name_condition.
    """
    if name_condition(box.name) and \
        len(box.insides) == 1 and \
        box.dom == box.cod == box.insides[0].dom == box.insides[0].cod:
        return box.insides[0]
    return box

def passive_to_active_voice(diagram):
    """
    Converts passive voice to active voice.
    """
    def remove_passive_frame(frame):
        if frame.name.lower() == "by" and \
            len(frame.insides) == 1 and \
            len(frame.insides[0].dom) == 1 and \
            len(frame.dom) == 2:
            name = spacy_model(frame.insides[0].name)[0].lemma_
            return Swap(Ty(frame.dom[0]), Ty(frame.dom[1])) \
                >> Box(name, frame.dom[::-1], frame.cod[::-1]) \
                >> Swap(Ty(frame.cod[1]), Ty(frame.cod[0]))
        return frame
    f = Functor(ob=lambda x: x, ar=lambda x: x, frame=remove_passive_frame)
    return f(diagram)
