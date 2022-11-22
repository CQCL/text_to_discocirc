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
    return expr.arg.expr_type == 'application' and\
            is_higher_order(expr.expr.final_type) and \
            not isinstance(expr.arg.arg.final_type, Func)

def if_lambda_pull_out(expr):
    return expr.arg.expr_type == 'lambda' and\
        is_higher_order(expr.expr.final_type) and\
        expr.expr.final_type.input == expr.arg.var.final_type and\
        expr.expr.final_type.output == expr.arg.var.final_type

def b_combinator(f, g, h):
    final_type = (h.final_type >> f.final_type.input) >>\
                 (h.final_type >> f.final_type.output)
    f = deepcopy(f)
    f.final_type = final_type
    return (f(g))(h)

def pull_out(expr):
    if expr.expr_type == 'application':
        if if_application_pull_out(expr):
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
            return f(g)
    return expr
