from discocirc.expr import Expr
from discocirc.helpers.closed import Func, uncurry_types

def expr_uncurry(expr):
    head = expr.head
    if expr.expr_type == "literal":
        new_expr = Expr.literal(expr.name,
                            uncurry_types(expr.typ, uncurry_everything=True))
    elif expr.expr_type == "lambda":
        if expr.body.expr_type == "lambda":
            # a -> b -> c = (a @ b) -> c
            a_b = Expr.lst([expr_uncurry(expr.var),
                            expr_uncurry(expr.body.var)])
            c = expr_uncurry(expr.body.body)
            new_expr = expr_uncurry(Expr.lmbda(a_b, c))
        else:
            var = expr_uncurry(expr.var)
            body = expr_uncurry(expr.body)
            new_expr = Expr.lmbda(var, body)
            if isinstance(body.typ, Func):
                new_expr.typ = var.typ @ body.typ.input >> body.typ.output

    elif expr.expr_type == "application":
        if expr.fun.expr_type == "application":
            a = expr_uncurry(expr.arg)
            b = expr_uncurry(expr.fun.arg)
            c = expr_uncurry(expr.fun.fun)
            interchange = all([isinstance(e.typ, Func) for e in [a, b]])
            a_b = Expr.lst([a, b], interchange=interchange)
            new_expr = expr_uncurry(c(a_b))
        else:
            arg = expr_uncurry(expr.arg)
            fun = expr_uncurry(expr.fun)
            new_expr = fun(arg)
    elif expr.expr_type == "list":
        expr_list = [expr_uncurry(e) for e in expr.expr_list]
        interchange = all([isinstance(e.typ, Func) for e in expr_list])
        new_expr = Expr.lst(expr_list, interchange=interchange)
    else:
        raise TypeError(f'Unknown type {expr.expr_type} of expression')
    new_expr.head = head
    return new_expr
