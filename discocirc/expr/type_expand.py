import random
import time
from discocirc.expr.expr_uncurry import expr_uncurry
from discocirc.expr.expr import Expr
from discocirc.helpers.closed import Func, Ty
from discocirc.helpers.discocirc_utils import apply_at_root, change_expr_typ, count_applications, expr_type_recursion


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
        return Expr.literal(expr.name, new_type, head=expr.head)
    else:
        return expr_type_recursion(expr, type_expand)

def expand_coordination(expr):
    if expr.expr_type == "application":
        head = expr.head
        fun = expand_coordination(expr.fun)
        expr = fun(expr.arg)
        for n in range(count_applications(expr, branch='arg'), 0, -1):
            nth_arg = expr
            funs = []
            heads = []
            for _ in range(n):
                funs.append(nth_arg.fun)
                heads.append(nth_arg.head)
                nth_arg = nth_arg.arg
            if nth_arg.typ == Ty('n') and nth_arg.head and len(nth_arg.head) > 1:
                var = Expr.literal(f"x_{random.randint(1000,9999)}", nth_arg.typ)
                var.head = nth_arg.head
                body = var
                for f, head in zip(reversed(funs), reversed(heads)):
                    f = expand_coordination(f)
                    body = f(body)
                    body.head = head
                composition = Expr.lmbda(var, body)
                nth_arg = change_expr_typ(nth_arg, composition.typ >> nth_arg.typ)
                nth_arg = apply_at_root(nth_arg, composition)
                expr = change_expr_typ(nth_arg, expr.typ)
                break
        expr.head = head
    return expr

def n_expand(expr):
    if expr.expr_type == "literal":
        new_type = expand_closed_type(expr.typ, Ty('n'))
        return change_expr_typ(expr, new_type)
    elif expr.expr_type == "application":
        head = expr.head
        if expr.arg.typ == Ty('n') and expr.arg.head:
            arg = n_expand(expr.arg)
            fun = expr.fun
            uncurried_arg = expr_uncurry(arg)
            old_uncurried_arg = expr_uncurry(expr.arg)
            if isinstance(uncurried_arg.typ, Func):
                arg_output_wires = len(uncurried_arg.typ.output)
                old_arg_output_wires = len(old_uncurried_arg.typ.output)
            else:
                arg_output_wires = len(uncurried_arg.typ) # find the number of output nouns of the argument
                old_arg_output_wires = len(old_uncurried_arg.typ) # find the number of output nouns of the argument
            if arg.head and old_arg_output_wires != arg_output_wires:
                wire_index = 0
                for e in expr_uncurry(arg).arg.expr_list:
                    if e.typ == Ty('n'):
                        if e.head == arg.head:
                            break
                        wire_index = wire_index + 1
                x = Expr.literal(f"temp_{str(time.time())[-4:]}", typ=Ty('n'))
                id_expr = Expr.lmbda(x, x)
                left_ids = [id_expr] * wire_index
                right_ids = [id_expr] * (arg_output_wires - wire_index - 1)
                fun = Expr.lst(left_ids + [fun] + right_ids)
                expr = fun(arg)
                return n_expand(expr)
        fun = n_expand(expr.fun)
        arg = n_expand(expr.arg)
        if isinstance(arg.typ, Func):
            if arg.typ != fun.typ.input:
                fun = change_expr_typ(fun, arg.typ >> fun.typ.output)
        # new_fun_type = arg.typ >> fun.typ.output
        # fun = change_expr_typ(fun, new_fun_type)
        expr = fun(arg)
        expr.head = head
        return expr
    else:
        return expr_type_recursion(expr, n_expand)
    return expr
