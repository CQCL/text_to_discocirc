from random import randint

from discocirc.expr.expr import Expr
from discocirc.expr.inverse_beta import inverse_beta
from discocirc.helpers.closed import Func, Ty
from discocirc.helpers.discocirc_utils import change_expr_typ, count_applications, expr_type_recursion, n_fold_c_combinator


def is_higher_order(typ):
    if not isinstance(typ, Func):
        return False
    return isinstance(typ.input, Func) \
           or typ.input == Ty('p') \
           or typ.input == Ty('s')

def if_application_pull_out(expr):
    return expr.expr_type == 'application'\
            and expr.arg.expr_type == 'application'\
            and is_higher_order(expr.fun.typ)\
            and not isinstance(expr.arg.arg.typ, Func)

def b_combinator(expr):
    f = expr.fun
    g = expr.arg.fun
    h = expr.arg.arg
    new_type = (h.typ >> f.typ.input) >> \
                 (h.typ >> f.typ.output)
    bf = change_expr_typ(f, new_type)
    return (bf(g))(h)

def pull_out_application(expr):
    f = _pull_out(expr.fun)
    g = _pull_out(expr.arg)
    expr = Expr.apply(f, g, reduce=False)
    if if_application_pull_out(expr):
        expr = _pull_out(b_combinator(expr))
    return expr

def _pull_out(expr):
    head = expr.head if hasattr(expr, 'head') else None
    if expr.expr_type == 'literal':
        return expr
    elif expr.expr_type == 'application':
        expr = pull_out_application(expr)
        for n in range(1, count_applications(expr.arg)): # we can only apply C combinator if we have at least two applications
            n_c_combi_expr = Expr.apply(expr.fun, n_fold_c_combinator(expr.arg, n), reduce=False)
            n_c_combi_expr_pulled = pull_out_application(n_c_combi_expr)
            if n_c_combi_expr_pulled != n_c_combi_expr: # check if something was pulled out
                expr = _pull_out(n_c_combi_expr_pulled)
                break
        new_expr = expr
        if head:
            new_expr.head = head
        return new_expr        
    return expr_type_recursion(expr, _pull_out)

def pull_out(expr):
    return _pull_out(inverse_beta(expr))
