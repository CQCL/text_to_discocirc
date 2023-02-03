from copy import deepcopy
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

def count_applications(expr):
    count = 0
    while expr.expr_type == "application":
        count += 1
        expr = expr.expr
    return count

def c_combinator(expr):
    f = expr.expr.expr
    y = expr.expr.arg
    x = expr.arg
    f = deepcopy(f)
    new_type = x.typ >> (y.typ >> f.typ.output.output)
    f = change_expr_typ(f, new_type)
    return (f(x))(y)

def n_fold_c_combinator(expression, n):
    expr = deepcopy(expression)
    if expr.expr_type != "application" or expr.expr.expr_type != "application":
        raise ValueError(f'cannot apply C combinator {n} > {0} times to:\n{expression}')
    args = []
    for i in range(n-1):
        args.append(expr.arg)
        expr = expr.expr
        if expr.expr.expr_type != "application":
            raise ValueError(f'cannot apply C combinator {n} > {i+1} times to:\n{expression}')
    expr = c_combinator(expr)
    for arg in reversed(args):
        expr = c_combinator(expr(arg))
    return expr

def inv_n_fold_c_combinator(expression, n):
    expr = deepcopy(expression)
    if expr.expr_type != "application" or expr.expr.expr_type != "application":
        return expr
    args = []
    for i in range(n):
        expr = c_combinator(expr)
        args.append(expr.arg)
        expr = expr.expr
        if expr.expr_type != "application":
            raise ValueError(f'cannot apply C combinator {n} > {i+1} times to:\n{expression}')
    for arg in reversed(args):
        expr = expr(arg)
    return expr

def apply_at_root(fun, arg):
    return inv_n_fold_c_combinator(fun(arg), count_applications(fun))
