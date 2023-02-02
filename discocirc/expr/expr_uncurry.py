from discocirc.expr import Expr
from discocirc.helpers.closed import uncurry_types


def expr_uncurry(expr):
    if expr.expr_type == "literal":
        return Expr.literal(expr.name,
                            uncurry_types(expr.typ, uncurry_everything=True))
    elif expr.expr_type == "lambda":
        if expr.expr.expr_type == "lambda":
            # a -> b -> c = (a @ b) -> c
            a_b = Expr.lst([expr_uncurry(expr.var),
                            expr_uncurry(expr.expr.var)])
            c = expr_uncurry(expr.expr.expr)
            return expr_uncurry(Expr.lmbda(a_b, c))
        else:
            return Expr.lmbda(expr_uncurry(expr.var),
                              expr_uncurry(expr.expr))
    elif expr.expr_type == "application":
        if expr.expr.expr_type == "application":
            a = expr_uncurry(expr.arg)
            b = expr_uncurry(expr.expr.arg)
            c = expr_uncurry(expr.expr.expr)
            return expr_uncurry(c(Expr.lst([a, b], interchange=False)))
        else:
            arg = expr_uncurry(expr.arg)
            expr = expr_uncurry(expr.expr)
            return expr(arg)
    elif expr.expr_type == "list":
        return Expr.lst([expr_uncurry(e) for e in expr.expr_list],
                        interchange=False)
    else:
        raise TypeError(f'Unknown type {expr.expr_type} of expression')