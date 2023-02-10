from copy import deepcopy

from discocirc.expr.expr import Expr
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

def expr_has_variable(expr, variable):
    if expr == variable:
        return True
    if expr.expr_type == "literal":
        return False
    elif expr.expr_type == "list":
        return any([expr_has_variable(e, variable) for e in expr.expr_list])
    elif expr.expr_type == "lambda":
        return any([expr_has_variable(expr.expr, variable),
                    expr_has_variable(expr.var, variable)])
    elif expr.expr_type == "application":
        return any([expr_has_variable(expr.arg, variable),
                    expr_has_variable(expr.fun, variable)])
    return False

def expr_has_variables(expr, variables):
    for variable in variables:
        if expr_has_variable(expr, variable):
            return True
    return False

def get_args_to_pull(expr, variables):
    if expr in variables:
        return []
    if not isinstance(expr.typ, Func) and not expr_has_variables(expr, variables):
        return [expr]
    if expr.expr_type == "list":
        args_to_pull = []
        for e in expr.expr_list:
            args_to_pull.extend(get_args_to_pull(e, variables))
        return args_to_pull
    elif expr.expr_type == "lambda":
        return get_args_to_pull(expr.expr, variables + [expr.var])
    elif expr.expr_type == "application":
        return get_args_to_pull(expr.arg, variables) + \
               get_args_to_pull(expr.fun, variables)
    return []

def b_combinator(expr):
    f = expr.fun
    g = expr.arg.fun
    h = expr.arg.arg
    new_type = (h.typ >> f.typ.input) >> \
                 (h.typ >> f.typ.output)
    bf = change_expr_typ(f, new_type)
    return (bf(g))(h)

def pull_out_application(expr):
    f = pull_out(expr.fun)
    g = pull_out(expr.arg)
    expr = Expr.apply(f, g, reduce=False)
    if if_application_pull_out(expr):
        expr = pull_out(b_combinator(expr))
    return expr

def pull_out_lambda(expr):
    original_expr = deepcopy(expr)
    lambda_expr = expr.arg
    variables = []
    while lambda_expr.expr_type == 'lambda':
        variables.append(lambda_expr.var)
        lambda_expr = lambda_expr.expr
    lambda_expr = pull_out(lambda_expr)
    args_to_pull = get_args_to_pull(lambda_expr, variables)
    # we add args_to_pull to the list of lambda variables as
    # we are pulling out those args outside the lambda by
    # performing an inverse beta reduction.
    for variable in list(reversed(variables)) + args_to_pull:
        lambda_expr = Expr.lmbda(variable, lambda_expr)
    for arg in reversed(args_to_pull):
        lambda_expr = Expr.apply(lambda_expr, arg, reduce=False)
    expr = original_expr.fun(lambda_expr)
    if expr != original_expr:
        expr = pull_out(expr)
    return expr

def pull_out(expr):
    head = expr.head if hasattr(expr, 'head') else None
    if expr.expr_type == 'literal':
        return expr
    elif expr.expr_type == 'application':
        if expr.arg.expr_type == 'lambda' \
            and is_higher_order(expr.fun.typ):
            expr = pull_out_lambda(expr)
            new_expr = expr
        else:
            expr = pull_out_application(expr)
            for n in range(1, count_applications(expr.arg)): # we can only apply C combinator if we have at least two applications
                n_c_combi_expr = expr.fun(n_fold_c_combinator(expr.arg, n))
                n_c_combi_expr_pulled = pull_out_application(n_c_combi_expr)
                if n_c_combi_expr_pulled != n_c_combi_expr: # check if something was pulled out
                    expr = pull_out(n_c_combi_expr_pulled)
                    break
            new_expr = expr
        if head:
            new_expr.head = head
        return new_expr        
    return expr_type_recursion(expr, pull_out)

