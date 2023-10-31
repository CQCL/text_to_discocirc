from discocirc.expr.expr import Expr, expr_type_recursion
from discocirc.expr.uncurry import uncurry
from discocirc.expr.s_type_expand import expand_closed_type
from discocirc.helpers.closed import Func, Ty
from discocirc.helpers.discocirc_utils import change_expr_typ, create_lambda_swap, create_random_variable

def n_type_expand(expr):
    """
    Takes in an expr,
    expands n-type wires into several n-type wires as required
    """
    if expr.expr_type == "literal":
        # n-expanded types percolate down from the literals
        new_type = expand_closed_type(expr.typ, Ty('n'))
        return change_expr_typ(expr, new_type)
    elif expr.expr_type == "application":
        if expr.arg.typ == Ty('n') and expr.arg.head:
            # n-expansion possibly occurs
            return expand_app_with_n_arg(expr)
        # basically the usual recursion, except sometimes there is type mismatch
        return expand_app(expr)
    else:
        return expr_type_recursion(expr, n_type_expand)

def expand_app_with_n_arg(expr):
    """
    This is used in the case when expr is of the form f(g)
    and the expr g has type n

    The nontrivial case occurs when the output of g is 
    expanded into several n-type wires
    """
    arg = n_type_expand(expr.arg)
    num_arg_outputs = get_num_output_wires(arg)
    num_old_arg_outputs = get_num_output_wires(expr.arg)
    # check if n-type argument got expanded into multiple n's
    if num_old_arg_outputs != num_arg_outputs:
        wire_index = get_wire_index_of_head(arg)
        # slightly hacky solution: swap head to left and change wire_index to 0
        if wire_index != 0:
            # apply swaps so head noun of arg is on the left
            arg = swap_head_to_left(arg, wire_index)
        wire_index = 0 # this means left_ids is always empty
        left_ids = []
        right_ids = []
        # for i in range(0, wire_index):
        #     x = create_random_variable(arg.typ[i])
        #     id_expr = Expr.lmbda(x,x)
        #     left_ids.append(id_expr)
        for i in range(wire_index+1,len(arg.typ)):
            x = create_random_variable(arg.typ[i])
            id_expr = Expr.lmbda(x,x)
            right_ids.append(id_expr)
        fun = Expr.lst(left_ids + [expr.fun] + right_ids)
        new_expr = fun(arg)
        new_expr.head = expr.head
        return n_type_expand(new_expr)
    else:
        return expand_app(expr)

def expand_app(expr):
    """
    Performs expansion on an expr, in the case where expr is a basic application type
    This is almost a trivial recursion, except for one special case

    The special case if statement is triggered when the 'arg' is a Func type, e.g. a box, and
    the 'fun' is therefore a frame, and
    the 'arg' no longer fits in the frame after it gets n-expanded

    e.g. this occurs in cases such as 
        'Alice and Bob who like beer walked'
    Here, if n expansion is done before s expansion, then
    n-expansion causes an 'arg' to change from type nxn->s to nxn->sxn
    """
    fun = n_type_expand(expr.fun)
    arg = n_type_expand(expr.arg)
    if isinstance(arg.typ, Func):
        try:
            new_expr = fun(arg)
        except TypeError:
            fun = change_expr_typ(fun, arg.typ >> fun.typ.output)
            new_expr = fun(arg)
    else:
        new_expr = fun(arg)
    new_expr.head = expr.head
    return new_expr

def get_num_output_wires(expr):
    """
    Find the number of output wires of an expr
    """
    typ = uncurry(expr).typ
    if isinstance(typ, Func):
        return len(typ.output)
    else:
        return len(typ)

def get_wire_index_of_head(expr):
    """
    Find the index of the noun wire output corresponding to the head of the expr
    """
    wire_index = 0
    for e in uncurry(expr).arg.expr_list:
        if e.typ == Ty('n'):
            if e.head == expr.head:
                break
            wire_index = wire_index + 1
    return wire_index

def swap_head_to_left(expr, head_index):
    """
    We may assume expr.typ is some product of n's
    """    
    # generate permutation lambda-term that swaps head to left
    perm_list = list(range(len(expr.typ)))
    perm_list.insert(0, perm_list.pop(head_index))
    swap = create_lambda_swap(perm_list, expr.typ)

    # TODO: do the inverse swap to the top of the expr, so that
    # the top & bottom ordering of the noun wires remains consistent

    # TODO: generate the inverse swap lambda term
    # TODO: extract nouns from expr
    
    # compose w/ swap(s)
    expr = swap(expr)
    
    # TODO: reattach nouns to expr
    return expr
