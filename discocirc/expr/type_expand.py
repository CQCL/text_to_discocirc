from discocirc.expr.expr import Expr
from discocirc.helpers.closed import Func, Ty
from discocirc.helpers.discocirc_utils import expr_type_recursion


def expand_closed_type(typ, expand_which_type):
    if not isinstance(typ, Func):
        return typ
    args = []
    while isinstance(typ, Func):
        args.append(typ.input)
        typ = typ.output
    if typ == expand_which_type:
        n_nouns = sum([1 for i in Ty().tensor(*args) if Ty(i) == Ty('n')])
        typ = Ty().tensor(*([Ty('n')] * n_nouns))
    for arg in reversed(args):
        typ = expand_closed_type(arg, expand_which_type) >> typ
    return typ

def type_expand(expr):
    if expr.expr_type == "literal":
        new_type = expand_closed_type(expr.typ, Ty('s'))
        return Expr.literal(expr.name, new_type)
    else:
        return expr_type_recursion(expr, type_expand)
