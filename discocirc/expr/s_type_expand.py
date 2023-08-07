from discocirc.expr.expr import Expr, expr_type_recursion
from discocirc.helpers.closed import Func, Ty, uncurry_types
from discocirc.helpers.discocirc_utils import add_indices_to_types, create_lambda_swap, create_random_variable


def expand_closed_type(typ, expand_which_type):
    """
    Takes in a type, and replaces it with an appropriately expanded type
    expand_which_type in practice is set to either Ty('s') or Ty('p')
    """
    if not isinstance(typ, Func):
        if len(typ) > 1:
            typ = Ty().tensor(*[expand_closed_type(t, expand_which_type) for t in typ])
        return typ
    args = []
    indices = []
    # extract all the inputs from the type
    while isinstance(typ, Func):
        args.append(typ.input)
        indices.append(typ.index)
        typ = typ.output
    noun_args = reversed([i for i in args if i == Ty('n')])
    if typ == expand_which_type:
        typ = Ty().tensor(*noun_args)
    elif len(typ) > 1 and expand_which_type != Ty('n'):
        nouns_in_output = [t.index for t in typ if t == Ty('n')]
        new_typ = Ty()
        for t in typ:
            if Ty(t) == expand_which_type:
                nouns_not_accounted_for = [n for n in reversed(args) if n.index not in nouns_in_output]
                new_typ = new_typ @ Ty().tensor(*nouns_not_accounted_for)
            else:
                t = Ty(t) if not isinstance(t, Func) else t
                new_typ = new_typ @ t
        typ = new_typ
    for arg, index in zip(reversed(args), reversed(indices)):
        typ = expand_closed_type(arg, expand_which_type) >> typ
        typ.index = index
    return typ

def type_expand(expr, which_type):
    """
    Takes an expr, and applies the appropriate type expansion.

    which_type in practice is set to either Ty('s') or Ty('p')

    A special case is dealt with by the second if statement.
    It is triggered when we apply type expansion to a box that has swaps at the top.
    Then, the output wire, when expanded into several n-wires, must be composed 
    with the corresponding inverse swap.
    If this is not done, the indices of the input and output wires will not match.
    """
    if expr.expr_type == "literal":
        new_type = expand_closed_type(expr.typ, which_type)
        return Expr.literal(expr.name, new_type, head=expr.head)
    elif expr.expr_type == "application":
        arg = type_expand(expr.arg, which_type)
        orig_types = arg.typ
        new_types = expand_closed_type(expr.arg.typ, which_type)
        if add_indices_to_types(orig_types) != add_indices_to_types(new_types):
            # if types do not match, need to compose arg w/ appropriate swap
            orig_types = uncurry_types(orig_types, uncurry_everything=True)
            new_types = uncurry_types(new_types, uncurry_everything=True)
            assert orig_types.input == new_types.input
            input_type_indices = [typ.index for typ in orig_types.output]
            output_type_indices = [typ.index for typ in new_types.output]
            perm = [output_type_indices.index(idx) for idx in
                    input_type_indices]
            swap = create_lambda_swap(perm, orig_types.output)
            temp_vars = []
            for typ in reversed(list(orig_types.input)):
                temp_vars.append(create_random_variable(typ)) #TODO: don't hardcode type
                arg = arg(temp_vars[-1])
            arg = swap(arg)
            for temp_var in reversed(temp_vars):
                arg = Expr.lmbda(temp_var, arg)
        fun = type_expand(expr.fun, which_type)
        return fun(arg)
    else:
        return expr_type_recursion(expr, type_expand, which_type=which_type)

def s_type_expand(expr):
    return type_expand(expr, Ty('s'))

def p_type_expand(expr):
    """
    Currently this code treats p types exactly like s types

    This is not always desirable in reality, as some instances of p types
    should instead be treated like n types, and expanded analogously to n-type expansion
    """
    return type_expand(expr, Ty('p'))
