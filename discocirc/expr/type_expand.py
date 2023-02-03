from copy import deepcopy
import time
from discocirc.expr import expr_uncurry
from discocirc.expr.expr import Expr
from discocirc.helpers.closed import Func, Ty
from discocirc.helpers.discocirc_utils import change_expr_typ, expr_type_recursion


def expand_closed_type(typ, expand_which_type):
    if not isinstance(typ, Func):
        return typ
    args = []
    while isinstance(typ, Func):
        args.append(typ.input)
        typ = typ.output
    if typ == expand_which_type:
        n_nouns = sum([1 for i in Ty().tensor(*args) if Ty(i) == Ty('n')])
        typ = Ty().tensor(*([Ty('n')] * n_nouns))
    for arg in reversed(args):
        typ = expand_closed_type(arg, expand_which_type) >> typ
    return typ

def type_expand(expr):
    if expr.expr_type == "literal":
        new_type = expand_closed_type(expr.typ, Ty('s'))
        return Expr.literal(expr.name, new_type)
    else:
        return expr_type_recursion(expr, type_expand)


def n_expand(expr):
    if expr.expr_type == "literal":
        new_type = expand_closed_type(expr.typ, Ty('n'))
        expr_copy = deepcopy(expr)
        return change_expr_typ(expr_copy, new_type)
    elif expr.expr_type == "application":
        if expr.expr.typ.input == Ty('n'):
            arg = n_expand(expr.arg)
            fun = deepcopy(expr.expr)
            if hasattr(arg, 'head'):
                new_fun_type = Ty().tensor(*([Ty('n')] * len(arg.head))) \
                    >> expr.expr.typ.output
                fun = change_expr_typ(fun, new_fun_type)
            fun = n_expand(fun)
            uncurried_arg = expr_uncurry(arg)
            if isinstance(uncurried_arg.typ, Func):
                arg_output_wires = len(uncurried_arg.typ.output)
            else:
                arg_output_wires = len(uncurried_arg.typ)
            if hasattr(arg, 'head') and len(arg.head) < arg_output_wires:
                id_exprs = []
                for _ in range(arg_output_wires - len(arg.head)):
                    x = Expr.literal(f"temp__{time.time()}__", typ=Ty('n'))
                    id_exprs.append(Expr.lmbda(x, x))
                fun = Expr.lst([fun] + id_exprs)
            new_expr = fun(arg) # this won't work
        else:
            arg = n_expand(expr.arg)
            fun = n_expand(expr.expr)
            fun = deepcopy(fun)
            new_fun_type = arg.typ >> fun.typ.output
            fun = change_expr_typ(fun, new_fun_type)
            new_expr = fun(arg)
        if hasattr(expr, 'head'):
            new_expr.head = expr.head
        return new_expr
    return expr
