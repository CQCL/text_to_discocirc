
from discocirc.expr.expr import Expr, expr_type_recursion
from discocirc.helpers.closed import Func
from discocirc.helpers.discocirc_utils import create_random_variable


def expr_has_variable(expr, variable):
    """
    Checks if a given expr contains a certain variable.
    Does this recursively
    """
    if expr == variable:
        return True
    if expr.expr_type == "literal":
        return False
    elif expr.expr_type == "list":
        return any([expr_has_variable(e, variable) for e in expr.expr_list])
    elif expr.expr_type == "lambda":
        return any([expr_has_variable(expr.body, variable),
                    expr_has_variable(expr.var, variable)])
    elif expr.expr_type == "application":
        return any([expr_has_variable(expr.arg, variable),
                    expr_has_variable(expr.fun, variable)])
    return False

def expr_has_variables(expr, variables):
    """
    Checks if a given expr contains any of the variables in a given list of variables
    """
    for variable in variables:
        if expr_has_variable(expr, variable):
            return True
    return False

def remove_free_vars(expr, variables):
    """
    Takes in an expr, and replaces all non-function-type subexprs with dummy variables
    (unless the non-function-type subexpr contains variables from 'variables' input)

    Importantly, this procedure preserves the type and structure of the main expr

    Returns three items: 
        a list of the extracted free variables, 
        a corresponding list of the dummy variables they were replaced with,
        the modified expression (containing dummy variables)
    """
    if expr in variables:
        return [], [], expr
    elif not isinstance(expr.typ, Func) and not expr_has_variables(expr, variables):
        temp_var = create_random_variable(expr.typ, head=expr.head)
        return [expr], [temp_var], temp_var
    elif expr.expr_type == "list":
        free_vars = []
        bound_vars = []
        exprs = []
        for e in expr.expr_list:
            result = remove_free_vars(e, variables)
            free_vars.extend(result[0])
            bound_vars.extend(result[1])
            exprs.append(result[2])
        return free_vars, bound_vars, Expr.lst(exprs, head=expr.head)
    elif expr.expr_type == "lambda":
        if expr.var.expr_type == 'list':
            new_body = remove_free_vars(expr.body, variables + list(expr.var.expr_list))
        else:
            new_body = remove_free_vars(expr.body, variables + [expr.var])
        return new_body[0], new_body[1], Expr.lmbda(expr.var, new_body[2], head=expr.head)
    elif expr.expr_type == "application":
        arg_result = remove_free_vars(expr.arg, variables)
        fun_result = remove_free_vars(expr.fun, variables)

        return arg_result[0] + fun_result[0], \
               arg_result[1] + fun_result[1], \
            Expr.apply(fun_result[2], arg_result[2], reduce=False, head=expr.head)
    return [], [], expr

def inverse_beta(expr):
    """
    A technical step we do as part of the 'pulling out' procedure.
    Specifically, we apply 'inverse beta' on the entire expr, before
    performing _pull_out on the expr.

    inverse_beta allows us to 'pull out' even in the case when we have
    expressions containing the lambda abstraction type.

    Specifically, if we have an expr 
        λx.M
    where M is a term containing a free variable f, inverse_beta replaces this expr with 
        ( λy.λx.M[y/f] ) f
    i.e. it replaces instances of f with a dummy variable y, and brings 
    f itself outside the lambda scope
    """
    if expr.expr_type == 'literal':
        return expr
    # the 'lambda' case below is the only nontrivial case
    elif expr.expr_type == 'lambda':
        new_body = inverse_beta(expr.body) #recurse first
        expr = Expr.lmbda(expr.var, new_body, head=expr.head, index=expr.typ.index)
        variables = []
        flattened_variables = []
        indices = []
        while expr.expr_type == 'lambda':
            if expr.var.expr_type == 'list':
                for v in reversed(expr.var.expr_list):
                    flattened_variables.append(v)
            else:
                flattened_variables.append(expr.var)
            variables.append(expr.var)
            indices.append(expr.typ.index)
            expr = expr.body
        free_vars, bound_vars, expr = remove_free_vars(expr, flattened_variables)
        for variable, index in zip(list(reversed(variables)) + bound_vars, list(reversed(indices)) + [None]*len(bound_vars)):
            expr = Expr.lmbda(variable, expr, index=index)
        for arg in reversed(free_vars):
            expr = Expr.apply(expr, arg, reduce=False)
        return expr
    elif expr.expr_type == "application":
        arg = inverse_beta(expr.arg)
        fun = inverse_beta(expr.fun)
        new_expr = Expr.apply(fun, arg, reduce=False)
        if hasattr(expr, 'head'):
            new_expr.head = expr.head
        return new_expr
    return expr_type_recursion(expr, inverse_beta)
