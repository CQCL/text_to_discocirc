from discopy.biclosed import Under, Over
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


def type_check_term(term):
    """
    Given a term, check if all the arguments match the required ccg.

    :param term: Term - The term which should be type checked.
    :return: None - If term does not type check.
        ccg - The output type of the term, if it type checks.
    """
    ccg = term.ccg
    for arg in term.args:
        if not get_ccg_input(ccg) == type_check_term(arg):
            return None
        ccg = get_ccg_output(ccg)

    return ccg


def get_ccg_input(ccg):
    if isinstance(ccg, Under):
        return ccg.left
    elif isinstance(ccg, Over):
        return ccg.right


def get_ccg_output(ccg):
    if isinstance(ccg, Under):
        return ccg.right
    elif isinstance(ccg, Over):
        return ccg.left

