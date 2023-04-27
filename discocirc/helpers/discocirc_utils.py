from argparse import ArgumentError
from copy import deepcopy
import random
from discopy import rigid
from discopy.monoidal import Functor
from discopy import Ob

from discocirc.expr.expr import Expr, expr_type_recursion
from discocirc.helpers.closed import Func, Ty


def get_last_initial_noun(circ):
    """
    takes in a circuit with some number of states as the initial boxes
    returns the index of the last of these initial states
    """
    assert(circ.boxes[0].dom == rigid.Ty())   # check that the first box is a noun
    for i in range(len(circ.boxes) - 1):
        if circ.boxes[i].dom == rigid.Ty() and circ.boxes[i + 1].dom != rigid.Ty():
            return i
    # I think we only reach here if the entire circuit consists of nouns
    return len(circ.boxes)-1


def get_star_removal_functor():
    def star_removal_ob(ty):
        return rigid.Ty() if ty.name == "*" else ty
    def star_removal_ar(box):
        return rigid.Box(box.name, f(box.dom), f(box.cod))
    f = Functor(ob=star_removal_ob, ar=star_removal_ar)
    return f

def change_expr_typ(expr, new_type):
    expr = deepcopy(expr)
    if expr.expr_type == 'literal':
        expr.typ = new_type
        return expr
    elif expr.expr_type == 'application':
        fun_new_type = expr.arg.typ >> new_type
        fun = change_expr_typ(expr.fun, fun_new_type)
        new_expr = fun(expr.arg)
        return new_expr
    elif expr.expr_type == 'lambda':
        # TODO: below is not quite correct - all instances of the bound variable
        # inside the lambda 'body' also need to have their types changed 
        new_var = change_expr_typ(expr.var, new_type.input)
        new_expr = change_expr_typ(expr.body, new_type.output)
        return Expr.lmbda(new_var, new_expr)
    elif expr.expr_type == 'list':
        raise NotImplementedError("List type changing")
    #TODO: implement list case - a little bit more difficult as it is not clear,
    # which list element should be updated how


def count_applications(expr, branch='fun'):
    count = 0
    while expr.expr_type == "application":
        count += 1
        if branch == 'fun':
            expr = expr.fun
        elif branch == 'arg':
            expr = expr.arg
        else:
            raise ArgumentError(f'Invalid branch {branch}')
    return count

def c_combinator(expr):
    f = expr.fun.fun
    y = expr.fun.arg
    x = expr.arg
    new_type = x.typ >> (y.typ >> f.typ.output.output)
    f = change_expr_typ(f, new_type)
    return (f(x))(y)

def n_fold_c_combinator(expr, n):
    """
    given an expr with at least n+1 applications along the 'fun' branch,
    say f(a1)(a2)...(a(n+1)),
    we pass a1 down to a(n+1), i.e.
    return f'(a2)(a3)...(a(n+1))(a1)
    """
    if n == 0:
        return expr
    # check that expr has at least n+1 applications along the 'fun' branch
    max_c_combi = count_applications(expr, branch='fun')-1
    if n > max_c_combi:
        raise ValueError(f'cannot apply C combinator {n} > {max_c_combi} times to:\n{expr}')
    args = []
    for i in range(n-1):
        args.append(expr.arg)
        expr = expr.fun
    expr = c_combinator(expr)
    for arg in reversed(args):
        expr = c_combinator(expr(arg))
    return expr

def inv_n_fold_c_combinator(expr, n):
    """
    given an expr with at least n+1 applications along the 'fun' branch,
    say f(a1)(a2)...(a(n+1)),
    we pass a(n+1) up to a1, i.e.
    return f'(a(n+1))(a1)(a2)...(an)
    """
    # check that expr has at least n+1 applications along the 'fun' branch
    max_c_combi = count_applications(expr, branch='fun')-1
    if n > max_c_combi:
        raise ValueError(f'cannot apply C combinator {n} > {max_c_combi} times to:\n{expr}')
    args = []
    for i in range(n):
        expr = c_combinator(expr)
        args.append(expr.arg)
        expr = expr.fun
    for arg in reversed(args):
        expr = expr(arg)
    return expr

def apply_at_root(fun, arg):
    return inv_n_fold_c_combinator(fun(arg), count_applications(fun))

def create_random_variable(typ, head=None):
    return Expr.literal(f"x_{random.randint(1000,9999)}", typ=typ, head=head)

def add_indices_to_types(typ):
    if isinstance(typ, Func):
        return Func(add_indices_to_types(typ.input),
                    add_indices_to_types(typ.output),
                    typ.index)
    if len(typ.objects) == 1:
        obj = typ.objects[0]
        return Ty(f"{obj.name}[{typ.index}]", index=typ.index)
    return Ty(*[add_indices_to_types(x) for x in typ.objects], index=typ.index)

def expr_add_indices_to_types(expr):
    if expr.expr_type == 'literal':
        new_expr = deepcopy(expr)
        new_expr.typ = add_indices_to_types(expr.typ)
        return new_expr
    return expr_type_recursion(expr, expr_add_indices_to_types)
