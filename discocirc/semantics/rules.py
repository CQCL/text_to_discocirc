import spacy
from discopy.monoidal import Id, Box, Swap, Ty, Functor, Diagram

spacy_model = spacy.load('en_core_web_trf')


def remove_articles_ar(box):
    str = box.name.lower()
    if (str == "the" or str == "a" or str == "an") and \
        box.dom == box.cod and len(box.dom) == 1:
        return Id(box.dom)
    return box

remove_articles = Functor(ob=lambda x: x, ar=remove_articles_ar)


class RemoveToBeRewrite(Diagram):
    """
    Removes "am", "is", "are", "was", "were" from a diagram.
    """
    def frame_factory(name, dom, cod, insides):
        nm = name.lower()
        if nm =="am" or nm == "is" or nm == "are" or nm == "was" or nm == "were":
            assert dom == cod and len(insides) == 1
            return Id.tensor(*insides)
        return Diagram.frame_factory(name, dom, cod, insides)

remove_to_be = Functor(ob=lambda x: x, ar=lambda x: x, ar_factory=RemoveToBeRewrite)


class RemoveRelativePronounsRewrite(Diagram):
    """
    Removes relative pronouns "who", "whom", "whose", "which", "that" from a diagram.
    """
    def frame_factory(name, dom, cod, insides):
        nm = name.lower()
        if nm == "who" or nm == "whom" or nm == "whose" or nm == "which" or nm == "that":
            assert dom == cod and len(insides) == 1
            return Id.tensor(*insides)
        return Diagram.frame_factory(name, dom, cod, insides)

remove_relative_pronouns = Functor(ob=lambda x: x, ar=lambda x: x, ar_factory=RemoveRelativePronounsRewrite)


class PassiveToActiveRewrite(Diagram):
    """
    Converts passive voice to active voice.
    """
    def frame_factory(name, dom, cod, insides):
        if name.lower() == "by" and \
            len(insides) == 1 and \
            len(insides[0].dom) == 1 and \
            len(dom) == 2:
            name = spacy_model(insides[0].name)[0].lemma_
            return Swap(Ty(dom[0]), Ty(dom[1])) \
                >> Box(name, dom[::-1], cod[::-1]) \
                >> Swap(Ty(cod[1]), Ty(cod[0]))
        return Diagram.frame_factory(name, dom, cod, insides)

passive_to_active_voice = Functor(ob=lambda x: x, ar=lambda x: x, ar_factory=PassiveToActiveRewrite)
