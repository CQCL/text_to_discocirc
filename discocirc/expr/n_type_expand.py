
from discocirc.expr.expr import Expr, expr_type_recursion
from discocirc.expr.uncurry import uncurry
from discocirc.expr.s_type_expand import expand_closed_type
from discocirc.helpers.closed import Func, Ty
from discocirc.helpers.discocirc_utils import change_expr_typ, create_random_variable

def n_type_expand(expr):
    """
    Takes in an expr,
    expands n-type wires into several n-type wires as required
    """
    if expr.expr_type == "literal":
        return expand_literal(expr)
    elif expr.expr_type == "application":
        if expr.arg.typ == Ty('n') and expr.arg.head:
            return expand_app_with_n_arg(expr)
        return expand_app(expr)
    else:
        return expr_type_recursion(expr, n_type_expand)

def expand_literal(expr):
    """
    Applies type expansion to literal exprs
    """
    new_type = expand_closed_type(expr.typ, Ty('n'))
    return change_expr_typ(expr, new_type)

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
    if num_old_arg_outputs != num_arg_outputs:
        wire_index = get_wire_index_of_head(arg)
        left_ids = []
        right_ids = []
        for i in range(0, wire_index):
            x = create_random_variable(arg.typ[i])
            id_expr = Expr.lmbda(x,x)
            left_ids.append(id_expr)
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

    Note: the if statement is triggered when the 'arg' is a Func type, e.g. a box, and
    the 'fun' is therefore a frame, and
    the 'arg' no longer fits in the frame after it gets n-expanded

    e.g. this occurs in complicated cases such as 
    'Alice , Bob and Claire who like beer and wine walked'
    Here, if n expansion is done before s expansion, then
    n-expansion causes an 'arg' to change from type nxnxn->s to nxnxn->sxnxn
    """
    fun = n_type_expand(expr.fun)
    arg = n_type_expand(expr.arg)
    if isinstance(arg.typ, Func) and arg.typ != fun.typ.input:
        fun = change_expr_typ(fun, arg.typ >> fun.typ.output)
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
