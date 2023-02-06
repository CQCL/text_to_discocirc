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
            and is_higher_order(expr.fun.typ)\
            and not isinstance(expr.arg.arg.typ, Func)

def b_combinator(expr):
    f = expr.fun
    g = expr.arg.fun
    h = expr.arg.arg
    new_type = (h.typ >> f.typ.input) >> \
                 (h.typ >> f.typ.output)
    bf = deepcopy(f)
    bf = change_expr_typ(bf, new_type)
    return (bf(g))(h)

def pull_out_application(expr):
    f = pull_out(expr.fun)
    g = pull_out(expr.arg)
    expr = Expr.apply(f, g, reduce=False)
    if if_application_pull_out(expr):
        expr = pull_out(b_combinator(expr))
    return expr

def pull_out(expr):
    if expr.expr_type == 'application':
        if expr.arg.expr_type == 'lambda' \
            and is_higher_order(expr.fun.typ):
            original_expr = deepcopy(expr)
            f = expr.fun
            g = expr.arg
            # save the current variables of the expression in a list
            variables = []
            while g.expr_type == 'lambda':
                variables.append(g.var)
                g = g.expr
            g = pull_out(g)
            args_to_pull = []
            for n in range(count_applications(g)): # we can only apply C combinator if we have at least two applications
                nth_arg = n_fold_c_combinator(g, n).arg
                if nth_arg in variables or isinstance(nth_arg.typ, Func):
                    continue
                args_to_pull.append(nth_arg)
            # reapply the variables of the original expression
            for variable in reversed(variables):
                g = Expr.lmbda(variable, g)
            # the following two loops perform inverse beta reduction to take the args_to_pull outside the lambda
            for arg in args_to_pull:
                g = Expr.lmbda(arg, g)
            for arg in reversed(args_to_pull):
                g = Expr.apply(g, arg, reduce=False)
            expr = f(g)
            if expr != original_expr:
                expr = pull_out(expr)
            return expr
        else:
            expr = pull_out_application(expr)
            for n in range(1, count_applications(expr.arg)): # we can only apply C combinator if we have at least two applications
                n_c_combi_expr = expr.fun(n_fold_c_combinator(expr.arg, n))
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
