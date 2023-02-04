from copy import deepcopy

from discocirc.expr.expr import Expr
from discocirc.helpers.closed import Func, Ty
from discocirc.helpers.discocirc_utils import change_expr_typ, count_applications, n_fold_c_combinator


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
    return expr.expr_type == 'application' \
            and expr.arg.expr_type == 'lambda' \
            and is_higher_order(expr.expr.typ) \
            and expr.arg.expr.expr_type == 'application' \
            and not isinstance(expr.arg.expr.arg.typ, Func)

def b_combinator(expr):
    f = expr.expr
    g = expr.arg.expr
    h = expr.arg.arg
    new_type = (h.typ >> f.typ.input) >> \
                 (h.typ >> f.typ.output)
    bf = deepcopy(f)
    bf = change_expr_typ(bf, new_type)
    return (bf(g))(h)

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
            f = expr.expr
            g = expr.arg
            arg_to_pull = g.expr.arg
            inv_beta_g = Expr.apply(Expr.lmbda(arg_to_pull, g), arg_to_pull, reduce=False)
            expr = f(inv_beta_g)
            expr = pull_out(b_combinator(expr))
            return expr
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
