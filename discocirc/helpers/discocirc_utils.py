from discopy.rigid import Ty, Box
from discopy.monoidal import Functor


def get_last_initial_noun(circ):
    """
    takes in a circuit with some number of states as the initial boxes
    returns the index of the last of these initial states
    """
    assert(circ.boxes[0].dom == Ty())

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

def change_expr_typ(expr, new_type):
    if expr.expr_type == 'literal':
        expr.typ = new_type
        return expr
    elif expr.expr_type == 'application':
        fun_new_type = expr.arg.typ >> new_type
        fun = change_expr_typ(expr.expr, fun_new_type)
        new_expr = fun(expr.arg)
        return new_expr
    #TODO: implement lambda case
