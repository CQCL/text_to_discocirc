from random import randint
from discocirc.expr.expr import Expr
from discocirc.helpers.closed import Func
from discocirc.helpers.discocirc_utils import expr_type_recursion

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

def remove_free_vars(expr, variables):
    if expr in variables:
        return [], [], expr
    if not isinstance(expr.typ, Func) and not expr_has_variables(expr, variables):
        temp_var = Expr.literal(f"x_{randint(1000,9999)}", expr.typ, head=expr.head)
        return [expr], [temp_var], temp_var
    if expr.expr_type == "list":
        free_vars = []
        bound_vars = []
        exprs = []
        for e in expr.expr_list:
            result = remove_free_vars(e, variables)
            free_vars.extend(result[0])
            bound_vars.extend(result[1])
            exprs.append(result[2])
        return free_vars, bound_vars, Expr.lst(exprs, interchange=False, head=expr.head)
    elif expr.expr_type == "lambda":
        return remove_free_vars(expr.expr, variables + [expr.var])
    elif expr.expr_type == "application":
        arg_result = remove_free_vars(expr.arg, variables)
        fun_result = remove_free_vars(expr.fun, variables)

        return arg_result[0] + fun_result[0], \
               arg_result[1] + fun_result[1], \
            Expr.apply(fun_result[2], arg_result[2], reduce=False)
    return [], [], expr

def inverse_beta(expr):
    if expr.expr_type == 'literal':
        return expr
    if expr.expr_type == 'lambda':
        new_body = inverse_beta(expr.expr)
        expr = Expr.lmbda(expr.var, new_body, head=expr.head)
        variables = []
        while expr.expr_type == 'lambda':
            variables.append(expr.var)
            expr = expr.expr
        free_vars, bound_vars, expr = remove_free_vars(expr, variables)
        for variable in list(reversed(variables)) + bound_vars:
            expr = Expr.lmbda(variable, expr)
        for arg in reversed(free_vars):
            expr = Expr.apply(expr, arg, reduce=False)
        return expr
    return expr_type_recursion(expr, inverse_beta)