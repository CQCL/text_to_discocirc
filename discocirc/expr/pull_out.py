from copy import deepcopy

from discocirc.expr.expr import Expr
from discocirc.helpers.closed import Func, Ty


def is_higher_order(typ):
    if not isinstance(typ, Func):
        return False
    return isinstance(typ.input, Func) \
           or typ.input == Ty('p') \
           or typ.input == Ty('s')

def if_application_pull_out(expr):
    return expr.arg.expr_type == 'application'\
            and is_higher_order(expr.expr.typ)\
            and not isinstance(expr.arg.arg.typ, Func)

def if_lambda_pull_out(expr):
    return expr.arg.expr_type == 'lambda' and \
           is_higher_order(expr.expr.typ) and \
           expr.expr.typ.input.input == expr.arg.var.typ and \
           expr.expr.typ.output.input == expr.arg.var.typ

def b_combinator(f, g, h):
    new_type = (h.typ >> f.typ.input) >> \
                 (h.typ >> f.typ.output)
    bf = deepcopy(f)
    bf.typ = new_type
    return (bf(g))(h)

def c_combinator(f, y, x, n=1):
    if f.expr_type == "application":
        fx = c_combinator(f.expr, f.arg, x, n+1)
        return fx(y)
    typ = f.typ
    typs = []
    for _ in range(n):
        typs.append(typ.input)
        typ = typ.output
    x_type = typ.input
    output = typ.output
    for t in typs:
        output = t >> output
    output = x_type >> output
    f = deepcopy(f)
    f.typ = output
    return (f(x))(y)

def pull_out(expr):
    if expr.expr_type == 'application':
        if if_application_pull_out(expr):
            if expr.expr.expr_type == 'application':
                return pull_out(c_combinator(expr.expr.expr,
                                             expr.expr.arg,
                                             expr.arg))
            else:
                f = pull_out(expr.expr)
                arg = pull_out(expr.arg)
                g = arg.expr
                h = arg.arg
                return pull_out(b_combinator(f, g, h))
        elif if_lambda_pull_out(expr):
            expr2 = deepcopy(expr.expr)
            expr2.typ.input = expr2.typ.input.output
            expr2.typ.output = expr2.typ.output.output
            expr2 = pull_out(expr2(expr.arg.expr))
            return Expr.lmbda(expr.arg.var, expr2)
        else:
            f = pull_out(expr.expr)
            g = pull_out(expr.arg)
            if if_application_pull_out(f(g)):
                return pull_out(f(g))
            return f(g)
    elif expr.expr_type == 'lambda':
        return Expr.lmbda(expr.var, pull_out(expr.expr))
    elif expr.expr_type == 'list':
        pulled_out_list = [pull_out(e) for e in expr.expr_list]
        return Expr.lst(pulled_out_list)
    elif expr.expr_type == 'literal':
        return expr
    return expr
