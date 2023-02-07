from discocirc.expr import Expr
from discocirc.helpers.closed import uncurry_types

def expr_uncurry(expr):
    head = expr.head
    if expr.expr_type == "literal":
        new_expr = Expr.literal(expr.name,
                            uncurry_types(expr.typ, uncurry_everything=True))
    elif expr.expr_type == "lambda":
        if expr.expr.expr_type == "lambda":
            # a -> b -> c = (a @ b) -> c
            a_b = Expr.lst([expr_uncurry(expr.var),
                            expr_uncurry(expr.expr.var)])
            c = expr_uncurry(expr.expr.expr)
            new_expr = expr_uncurry(Expr.lmbda(a_b, c))
        else:
            new_expr = Expr.lmbda(expr_uncurry(expr.var),
                              expr_uncurry(expr.expr))
    elif expr.expr_type == "application":
        if expr.fun.expr_type == "application":
            a = expr_uncurry(expr.arg)
            b = expr_uncurry(expr.fun.arg)
            c = expr_uncurry(expr.fun.fun)
            new_expr = expr_uncurry(c(Expr.lst([a, b], interchange=False)))
        else:
            arg = expr_uncurry(expr.arg)
            fun = expr_uncurry(expr.fun)
            new_expr = fun(arg)
    elif expr.expr_type == "list":
        new_expr =  Expr.lst([expr_uncurry(e) for e in expr.expr_list],
                              interchange=False)
    else:
        raise TypeError(f'Unknown type {expr.expr_type} of expression')
    new_expr.head = head
    return new_expr
