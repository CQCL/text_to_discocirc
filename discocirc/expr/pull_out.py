from copy import deepcopy

from discocirc.expr.expr import Expr
from discocirc.helpers.closed import Func, Ty
from discocirc.helpers.discocirc_utils import change_expr_typ


def is_higher_order(typ):
    if not isinstance(typ, Func):
        return False
    return isinstance(typ.input, Func) \
           or typ.input == Ty('p') \
           or typ.input == Ty('s')

def if_application_pull_out(expr):
    return expr.expr_type == 'application'\
            and expr.arg.expr_type == 'application'\
            and is_higher_order(expr.expr.typ)\
            and not isinstance(expr.arg.arg.typ, Func)

def if_lambda_pull_out(expr):
    return expr.arg.expr_type == 'lambda' and \
           is_higher_order(expr.expr.typ) and \
           expr.expr.typ.input.input == expr.arg.var.typ and \
           expr.expr.typ.output.input == expr.arg.var.typ

def b_combinator(expr):
    f = expr.expr
    g = expr.arg.expr
    h = expr.arg.arg
    new_type = (h.typ >> f.typ.input) >> \
                 (h.typ >> f.typ.output)
    bf = deepcopy(f)
    bf = change_expr_typ(bf, new_type)
    return (bf(g))(h)

def c_combinator(expr):
    f = expr.expr.expr
    y = expr.expr.arg
    x = expr.arg
    f = deepcopy(f)
    new_type = x.typ >> (y.typ >> f.typ.output.output)
    f = change_expr_typ(f, new_type)
    return (f(x))(y)

def count_applications(expr):
    count = 0
    while expr.expr_type == "application":
        count += 1
        expr = expr.expr
    return count

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

def pull_out_application(expr):
    f = pull_out(expr.expr)
    g = pull_out(expr.arg)
    expr = f(g)
    if if_application_pull_out(expr):
        expr = pull_out(b_combinator(expr))
    return expr

def pull_out(expr):
    if expr.expr_type == 'application':
        if if_lambda_pull_out(expr):
            expr2 = deepcopy(expr.expr)
            expr2.typ.input = expr2.typ.input.output
            expr2.typ.output = expr2.typ.output.output
            expr2 = pull_out(expr2(expr.arg.expr))
            return Expr.lmbda(expr.arg.var, expr2)
        else:
            expr = pull_out_application(expr)
            for n in range(1, count_applications(expr.arg)): # we can only apply C combinator if we have at least two applications
                n_c_combi_expr = expr.expr(n_fold_c_combinator(expr.arg, n))
                n_c_combi_expr_pulled = pull_out_application(n_c_combi_expr)
                if n_c_combi_expr_pulled != n_c_combi_expr: # check if something was pulled out
                    return pull_out(n_c_combi_expr_pulled)
            return expr
    elif expr.expr_type == 'lambda':
        return Expr.lmbda(expr.var, pull_out(expr.expr))
    elif expr.expr_type == 'list':
        pulled_out_list = [pull_out(e) for e in expr.expr_list]
        return Expr.lst(pulled_out_list)
    elif expr.expr_type == 'literal':
        return expr
    return expr
