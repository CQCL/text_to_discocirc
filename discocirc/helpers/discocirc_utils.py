from argparse import ArgumentError
from copy import deepcopy
from discopy import rigid
from discopy.rigid import Ty
from discopy.monoidal import Functor

from discocirc.expr.expr import Expr, expr_type_recursion
from discocirc.helpers.closed import Func, Ty, index_to_string


def get_last_initial_noun(circ):
    """
    takes in a circuit with some number of states as the initial boxes
    returns the index of the last of these initial states
    """
    assert(circ.boxes[0].dom == rigid.Ty())   # check that the first box is a noun
    for i in range(len(circ.boxes) - 1):
        if circ.boxes[i].dom == rigid.Ty() and circ.boxes[i + 1].dom != rigid.Ty():
            return i
    # I think we only reach here if the entire circuit consists of nouns
    return len(circ.boxes)-1


def get_star_removal_functor():
    """
    Get a functor, which removes all instances of the star type from a diagram.
    """
    def star_removal_ob(ty):
        return rigid.Ty() if ty.name == "*" else ty
    def star_removal_ar(box):
        return rigid.Box(box.name, f(box.dom), f(box.cod))
    f = Functor(ob=star_removal_ob, ar=star_removal_ar)
    return f

def change_expr_typ(expr, new_type):
    """
    Given an expr, we attempt to change its type to new_type

    This is nontrivial, because we need to recursively change
    the type of the expr so everything remains internally consistent

    This function is a little conceptually fraught right now.
    In particular, the 'lambda' case is wrong,
    and the 'list' case needs further thought in order to implement
    """
    expr = deepcopy(expr)
    if expr.expr_type == 'literal':
        expr.typ = new_type
        return expr
    elif expr.expr_type == 'application':
        fun_new_type = expr.arg.typ >> new_type
        fun = change_expr_typ(expr.fun, fun_new_type)
        new_expr = fun(expr.arg)
        return new_expr
    elif expr.expr_type == 'lambda':
        # TODO: below is not quite correct - all instances of the bound variable
        # inside the lambda 'body' also need to have their types changed 
        new_var = change_expr_typ(expr.var, new_type.input)
        new_expr = change_expr_typ(expr.body, new_type.output)
        return Expr.lmbda(new_var, new_expr)
    elif expr.expr_type == 'list':
        raise NotImplementedError("List type changing")
    #TODO: implement list case - a little bit more difficult as it is not clear,
    # which list element should be updated how


def count_applications(expr, branch='fun'):
    """
    Count the number of consecutive applications, when following either the
    'fun' or 'arg' branch of the expr tree.
    """
    count = 0
    while expr.expr_type == "application":
        count += 1
        if branch == 'fun':
            expr = expr.fun
        elif branch == 'arg':
            expr = expr.arg
        else:
            raise ArgumentError(f'Invalid branch {branch}')
    return count

def c_combinator(expr):
    """
    Given an expr of the form
        f(y)(x)
    Change the type of the subexpr f, to obtain a new expr f', say,
    such that we can return
        f'(x)(y)
    i.e. the order of the arguments y, x are swapped
    """
    f = expr.fun.fun
    y = expr.fun.arg
    x = expr.arg
    new_type = x.typ >> (y.typ >> f.typ.output.output)
    f = change_expr_typ(f, new_type)
    return (f(x))(y)

def n_fold_c_combinator(expr, n):
    """
    given an expr with at least n+1 applications along the 'fun' branch,
    say f(a1)(a2)...(a(n+1)),
    we pass a1 down to a(n+1), i.e.
    return f'(a2)(a3)...(a(n+1))(a1)
    """
    if n == 0:
        return expr
    # check that expr has at least n+1 applications along the 'fun' branch
    max_c_combi = count_applications(expr, branch='fun')-1
    if n > max_c_combi:
        raise ValueError(f'cannot apply C combinator {n} > {max_c_combi} times to:\n{expr}')
    args = []
    for i in range(n-1):
        args.append(expr.arg)
        expr = expr.fun
    expr = c_combinator(expr)
    for arg in reversed(args):
        expr = c_combinator(expr(arg))
    return expr

def inv_n_fold_c_combinator(expr, n):
    """
    given an expr with at least n+1 applications along the 'fun' branch,
    say f(a1)(a2)...(a(n+1)),
    we pass a(n+1) up to a1, i.e.
    return f'(a(n+1))(a1)(a2)...(an)
    """
    # check that expr has at least n+1 applications along the 'fun' branch
    max_c_combi = count_applications(expr, branch='fun')-1
    if n > max_c_combi:
        raise ValueError(f'cannot apply C combinator {n} > {max_c_combi} times to:\n{expr}')
    args = []
    for i in range(n):
        expr = c_combinator(expr)
        args.append(expr.arg)
        expr = expr.fun
    for arg in reversed(args):
        expr = expr(arg)
    return expr

def apply_at_root(fun, arg):
    """
    usually, if we apply a 'fun' to an 'arg', we just return 'fun(arg)'

    apply_at_root instead applies 'arg' and ensures it is the first argument that is applied

    e.g. if 'fun' is of the form 
        f(x)(y)(z),
    where f is not an instance of application,
    then apply_at_root(fun,arg) returns
        f(arg)(x)(y)(z)
    """
    return inv_n_fold_c_combinator(fun(arg), count_applications(fun))

def add_indices_to_types(typ):
    if isinstance(typ, Func):
        return Func(add_indices_to_types(typ.input),
                    add_indices_to_types(typ.output),
                    typ.index)
    if len(typ.inside) == 1:
        obj = typ.inside[0]
        return Ty(f"{obj.name}{index_to_string(typ.index)}", index=typ.index)
    return Ty(*[add_indices_to_types(x) for x in typ.inside], index=typ.index)

def expr_add_indices_to_types(expr):
    if expr.expr_type == 'literal':
        new_expr = deepcopy(expr)
        new_expr.typ = add_indices_to_types(expr.typ)
        return new_expr
    return expr_type_recursion(expr, expr_add_indices_to_types)


random_variable_counter = 0
def create_random_variable(typ, head=None):
    """
    Create an expr with a random variable name of type typ.
    To ensure that no two random variables have the same name, we keep a global
    counter.
    """
    global random_variable_counter
    random_variable_counter += 1
    return Expr.literal(f"x_{random_variable_counter}", typ=typ, head=head)

def create_lambda_swap(perm, input_types):
    """
    Given a list of integers, create a lambda expression that swaps the wires
    as specified by the list where the ith wire is swapped to all wires j where
    new_order[j] = i.

    :param perm: The list of integers specifying the new order of the wires.
    :param input_types: The types of the inputs of the swap
    :return: The lambda expression that swaps the wires.
    """
    assert len(perm) == len(input_types)
    temp_vars = []
    for typ in input_types:
        temp_vars.append(create_random_variable(typ))

    lst = [temp_vars[i] for i in perm]
    swap = Expr.lmbda(Expr.lst(temp_vars), Expr.lst(lst))
    return swap
