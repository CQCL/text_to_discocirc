from discocirc.expr import Expr
from discocirc.helpers.closed import Func, uncurry_types
from discocirc.helpers.discocirc_utils import create_random_variable

def uncurry(expr):
    """
    Returns an uncurried version of the given expression.
    """
    head = expr.head
    if expr.expr_type == "literal":
        new_expr = Expr.literal(expr.name,
                            uncurry_types(expr.typ, uncurry_everything=True))
    elif expr.expr_type == "lambda":
        new_var = uncurry(expr.var)
        new_body = uncurry(expr.body)
        if isinstance(new_body.typ, Func):
            if new_body.var.expr_type == "list":
                new_var_list = Expr.lst(new_body.var.expr_list + (new_var,),
                                        False,
                                        new_body.var.head)
                new_expr = Expr.lmbda(new_var_list, new_body.body)
            else:
                var2 = create_random_variable(new_body.typ.input)
                product_var = Expr.lst([var2, new_var], interchange=False)
                new_expr = Expr.lmbda(product_var, new_body(var2))
        else:
            new_expr = Expr.lmbda(new_var, new_body)
    elif expr.expr_type == "application":
        if expr.fun.expr_type == "application":
            a = uncurry(expr.arg)
            b = uncurry(expr.fun.arg)
            c = uncurry(expr.fun.fun)
            # I can't figure out when to interchange and when to not
            try:
                a_b = Expr.lst([a, b], interchange=False)
                new_expr = c(a_b)
            except TypeError:
                a_b = Expr.lst([a, b], interchange=True)
                new_expr = c(a_b)
            new_expr = uncurry(new_expr)
        else:
            arg = uncurry(expr.arg)
            fun = uncurry(expr.fun)
            new_expr = fun(arg)
    elif expr.expr_type == "list":
        new_expr = Expr.lst([uncurry(e) for e in expr.expr_list])
    else:
        raise TypeError(f'Unknown type {expr.expr_type} of expression')
    new_expr.head = head
    return new_expr
