from copy import deepcopy
from discocirc.closed import Func, Ty
from discocirc.expr import Expr


def is_higher_order(simple_type):
    if not isinstance(simple_type, Func):
        return False
    return isinstance(simple_type.input, Func) \
                or simple_type.input == Ty('p')\
                or simple_type.input == Ty('s')

def if_application_pull_out(expr):
    return expr.arg.expr_type == 'application'\
            and is_higher_order(expr.expr.final_type)\
            and not isinstance(expr.arg.arg.final_type, Func)

def if_lambda_pull_out(expr):
    return expr.arg.expr_type == 'lambda' and\
        is_higher_order(expr.expr.final_type) and\
        expr.expr.final_type.input.input == expr.arg.var.final_type and\
        expr.expr.final_type.output.input == expr.arg.var.final_type

def b_combinator(f, g, h):
    final_type = (h.final_type >> f.final_type.input) >>\
                 (h.final_type >> f.final_type.output)
    bf = deepcopy(f)
    bf.final_type = final_type
    return (bf(g))(h)

def c_combinator(f_y_x):
    f = deepcopy(f_y_x.expr.expr)
    y = f_y_x.expr.arg
    x = f_y_x.arg
    f.final_type = f.final_type.output.input >>\
        (f.final_type.input >> f.final_type.output.output)
    return (f(x))(y)

def pull_out(expr):
    if expr.expr_type == 'application':
        if if_application_pull_out(expr):
            if expr.expr.expr_type == 'application':
                return pull_out(c_combinator(expr))
            else:
                f = pull_out(expr.expr)
                arg = pull_out(expr.arg)
                g = arg.expr
                h = arg.arg
                return pull_out(b_combinator(f, g, h))
        elif if_lambda_pull_out(expr):
            expr2 = deepcopy(expr.expr)
            expr2.final_type.input = expr2.final_type.input.output
            expr2.final_type.output = expr2.final_type.output.output
            expr2 = pull_out(expr2(expr.arg.expr))
            return Expr.lmbda(expr.arg.var, expr2)
        else:
            f = pull_out(expr.expr)
            g = pull_out(expr.arg)
            if if_application_pull_out(f(g)):
                return pull_out(f(g))
            return f(g)
    elif expr.expr_type == 'lambda':
        return Expr.lmbda(expr.var, pull_out(expr.expr), expr.simple_type)
    elif expr.expr_type == 'list':
        pulled_out_list = [pull_out(e) for e in expr.expr_list]
        return Expr.lst(pulled_out_list, expr.simple_type)
    elif expr.expr_type == 'literal':
        return expr
    return expr
