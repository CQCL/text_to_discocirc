from discocirc.expr.expr import Expr
from discocirc.helpers.closed import Func, Ty


def s_expand_t(t):
    typs = []
    while isinstance(t, Func):
        typs.insert(0, t.input)
        t = t.output
    if t == Ty("s"):
        n_nouns = sum([1 for i in Ty().tensor(*typs) if Ty(i) == Ty('n')])
        t = Ty().tensor(*([Ty('n')] * n_nouns))
    for typ in typs:
        t = s_expand_t(typ) >> t
    return t


def do_the_obvious(expr, function):
    if expr.expr_type == "literal":
        new_expr = function(expr)
    elif expr.expr_type == "list":
        new_list = [function(e) for e in expr.expr_list]
        new_expr = Expr.lst(new_list)
    elif expr.expr_type == "lambda":
        new_expr = function(expr.expr)
        new_var = function(expr.var)
        new_expr = Expr.lmbda(new_var, new_expr)
    elif expr.expr_type == "application":
        arg = function(expr.arg)
        body = function(expr.expr)
        # TODO: is this alright? I don't think so, to be honest.
        #  If we fix unary rule, we can also fix this one
        body.final_type.input = arg.final_type
        new_expr = body(arg)
    else:
        raise TypeError(f'Unknown type {expr.expr_type} of expression')
    if hasattr(expr, 'head'):
        new_expr.head = expr.head
    return new_expr

def type_expand(expr):
    if expr.expr_type == "literal":
        final_type = s_expand_t(expr.final_type)
        return Expr.literal(expr.name, final_type)
    return do_the_obvious(expr, type_expand)