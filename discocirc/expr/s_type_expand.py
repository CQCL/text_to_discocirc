from discocirc.expr.expr import Expr, expr_type_recursion
from discocirc.helpers.closed import Func, Ty


def expand_closed_type(typ, expand_which_type):
    if not isinstance(typ, Func):
        return typ
    args = []
    indices = []
    while isinstance(typ, Func):
        args.append(typ.input)
        indices.append(typ.index)
        typ = typ.output
    n_nouns = sum([1 for i in Ty('').tensor(*args) if i == Ty('n')])
    noun_args = reversed([i for i in args if i == Ty('n')])
    if typ == expand_which_type:
        typ = Ty().tensor(*noun_args)
    elif len(typ) > 1 and expand_which_type != Ty('n'): #TODO coindexing in this case
        num_output_nouns = sum([1 for t in typ if t == Ty('n')])
        new_typ = Ty()
        for t in typ:
            if Ty(t) == expand_which_type:
                new_typ = new_typ @ Ty().tensor(*([Ty('n')] * (n_nouns - num_output_nouns)))
            else:
                t = Ty(t) if not isinstance(t, Func) else t
                new_typ = new_typ @ t
        typ = new_typ
    for arg, index in zip(reversed(args), reversed(indices)):
        typ = expand_closed_type(arg, expand_which_type) >> typ
        typ.index = index
    return typ

def s_type_expand(expr):
    if expr.expr_type == "literal":
        new_type = expand_closed_type(expr.typ, Ty('s'))
        return Expr.literal(expr.name, new_type, head=expr.head)
    else:
        return expr_type_recursion(expr, s_type_expand)

def p_type_expand(expr):
    if expr.expr_type == "literal":
        new_type = expand_closed_type(expr.typ, Ty('p'))
        return Expr.literal(expr.name, new_type, head=expr.head)
    else:
        return expr_type_recursion(expr, s_type_expand)
