import random
from discocirc.expr import Expr
from discocirc.helpers.closed import Func, uncurry_types
from discocirc.helpers.discocirc_utils import create_random_variable

def expr_uncurry(expr):
    head = expr.head
    if expr.expr_type == "literal":
        new_expr = Expr.literal(expr.name,
                            uncurry_types(expr.typ, uncurry_everything=True))
    elif expr.expr_type == "lambda":
        new_var = expr_uncurry(expr.var)
        new_body = expr_uncurry(expr.expr)
        if isinstance(new_body.typ, Func):
            var2 = create_random_variable(new_body.typ.input)
            product_var = Expr.lst([var2, new_var], interchange=False)
            new_expr = Expr.lmbda(product_var, new_body(var2))
        else:
            new_expr = Expr.lmbda(new_var, new_body)
    elif expr.expr_type == "application":
        if expr.fun.expr_type == "application":
            a = expr_uncurry(expr.arg)
            b = expr_uncurry(expr.fun.arg)
            c = expr_uncurry(expr.fun.fun)
            # I can't figure out when to interchange and when to not
            try:
                a_b = Expr.lst([a, b], interchange=False)
                new_expr = c(a_b)
            except TypeError:
                a_b = Expr.lst([a, b], interchange=True)
                new_expr = c(a_b)
            new_expr = expr_uncurry(new_expr)
        else:
            arg = expr_uncurry(expr.arg)
            fun = expr_uncurry(expr.fun)
            new_expr = fun(arg)
    elif expr.expr_type == "list":
        new_expr = Expr.lst([expr_uncurry(e) for e in expr.expr_list])
    else:
        raise TypeError(f'Unknown type {expr.expr_type} of expression')
    new_expr.head = head
    return new_expr
