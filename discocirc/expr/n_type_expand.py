
import random
from discocirc.expr.expr import Expr
from discocirc.expr.expr_uncurry import expr_uncurry
from discocirc.expr.s_type_expand import expand_closed_type
from discocirc.helpers.closed import Func, Ty
from discocirc.helpers.discocirc_utils import change_expr_typ, create_random_variable, expr_type_recursion


def n_type_expand(expr):
    if expr.expr_type == "literal":
        return expand_literal(expr)
    elif expr.expr_type == "application":
        if expr.arg.typ == Ty('n') and expr.arg.head:
            return expand_app_with_n_arg(expr)
        return expand_app(expr)
    else:
        return expr_type_recursion(expr, n_type_expand)

def expand_literal(expr):
    new_type = expand_closed_type(expr.typ, Ty('n'))
    return change_expr_typ(expr, new_type)

def expand_app_with_n_arg(expr):
    arg = n_type_expand(expr.arg)
    num_arg_outputs = get_num_output_wires(arg)
    num_old_arg_outputs = get_num_output_wires(expr.arg)
    if num_old_arg_outputs != num_arg_outputs:
        # TODO: We need to figure out swaps here
        wire_index = get_wire_index_of_head(arg)
        x = create_random_variable(Ty('n'))
        id_expr = Expr.lmbda(x, x)
        left_ids = [id_expr] * wire_index
        right_ids = [id_expr] * (num_arg_outputs - wire_index - 1)
        fun = Expr.lst(left_ids + [expr.fun] + right_ids)
        new_expr = fun(arg)
        new_expr.head = expr.head
        return n_type_expand(expr)
    else:
        return expand_app(expr)

def expand_app(expr):
    fun = n_type_expand(expr.fun)
    arg = n_type_expand(expr.arg)
    if isinstance(arg.typ, Func) and arg.typ != fun.typ.input:
        fun = change_expr_typ(fun, arg.typ >> fun.typ.output)
    new_expr = fun(arg)
    new_expr.head = expr.head
    return new_expr

def get_num_output_wires(expr):
    typ = expr_uncurry(expr).typ
    if isinstance(typ, Func):
        return len(typ.output)
    else:
        return len(typ)

def get_wire_index_of_head(expr):
    wire_index = 0
    for e in expr_uncurry(expr).arg.expr_list:
        if e.typ == Ty('n'):
            if e.head == expr.head:
                break
            wire_index = wire_index + 1
    return wire_index
