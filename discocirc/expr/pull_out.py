
from discocirc.expr.expr import Expr, expr_type_recursion
from discocirc.expr.inverse_beta import inverse_beta
from discocirc.helpers.closed import Func, Ty
from discocirc.helpers.discocirc_utils import change_expr_typ, count_applications, n_fold_c_combinator


def is_higher_order(typ):
    """
    Checks if a type is 'higher-order', hence corresponding to a frame
    
    It is considered higher-order if it is a
    Func type, and furthermore its input is either a
    s-type, p-type, or itself a Func type
    """
    if not isinstance(typ, Func):
        return False
    return isinstance(typ.input, Func) \
           or typ.input == Ty('p') \
           or typ.input == Ty('s')

def if_application_pull_out(expr):
    """
    Given an expr that is application type,
    checks if pulling out can/should be performed
    """
    return expr.expr_type == 'application'\
            and expr.arg.expr_type == 'application'\
            and is_higher_order(expr.fun.typ)\
            and not isinstance(expr.arg.arg.typ, Func)

def b_combinator(expr):
    """
    Input expr is of form
        f(g(h))
    We change the type of f, to get a new expression f' say,
    such that we can return an expr
        f'(g)(h)
    """
    f = expr.fun
    g = expr.arg.fun
    h = expr.arg.arg
    # Below we choose to set the inner index to expr.typ.index. Previously had it as g.typ.index
    new_type = Func(g.typ, Func(h.typ, f.typ.output, expr.typ.index), f.typ.index) 
    bf = change_expr_typ(f, new_type)
    bf_g = Expr.apply(bf, g, reduce=False)
    return Expr.apply(bf_g, h, reduce=False)

def _pull_out(expr):
    """
    The main part of the pull out routine
    """
    head = expr.head if hasattr(expr, 'head') else None
    typ_index = expr.typ.index
    if expr.expr_type == 'literal':
        return expr
    elif expr.expr_type == 'application':
        f = _pull_out(expr.fun)
        g = _pull_out(expr.arg)
        expr = Expr.apply(f, g, reduce=False)
        num_pulled = 0
        pulled_args = []
        for n in range(count_applications(expr.arg)):
            n_c_combi_expr = Expr.apply(expr.fun, n_fold_c_combinator(expr.arg, n-num_pulled), reduce=False)
            if if_application_pull_out(n_c_combi_expr):
                expr = b_combinator(n_c_combi_expr)
                pulled_args.append(expr.arg)
                expr = expr.fun
                num_pulled += 1
        for arg in reversed(pulled_args):
            expr = Expr.apply(expr, arg, reduce=False)
        expr.typ.index = typ_index
        expr.head = head
        return expr
    return expr_type_recursion(expr, _pull_out)

def pull_out(expr):
    """
    The full pull_out routine consists of doing an inverse_beta pass over the expr,
    before performing _pull_out

    inverse_beta puts the expression in such a form that we can also pull out when
    the expr contains lambda abstractions
    """
    return _pull_out(inverse_beta(expr))
