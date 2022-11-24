#%%
from copy import deepcopy

from lambeq import BobcatParser
from discocirc.closed import Func, Ty, uncurry_types
from discocirc.expr import Expr
from discocirc.sentence_to_circuit import make_term, make_diagram, sentence2circ
from discocirc.pulling_out import is_higher_order
from discocirc.term import Term
from discocirc.frame import Frame

#%%
parser = BobcatParser()
draw_term = lambda term: (Frame.get_decompose_functor())(make_diagram(term)).draw()

# # %%
# # sentence = "Alice quickly passionately gives Bob flowers"
# sentence = "Alice and Bob quickly passionately give flowers to Claire"
# # sentence = "Alice quickly gives Claire red smelly fish"
# diagram = parser.sentence2tree(sentence).to_biclosed_diagram()
# term = make_term(diagram)

# %%
def just(term):
    return Term(term.name, term.simple_type, term.simple_type, [])


def pop(term):
    args, h = term.args[:-1], term.args[-1]
    final_type = h.final_type >> term.final_type
    g = Term(term.name, term.simple_type, final_type, args)
    assert term == g(h)
    return g, h


def b_combinator(f, A):
    """
    This function models the operation of the B-combinator on f,
    which is dependent on the type A.
    We have B = λ f: (B -> C). λ g: (A -> B). λ h: A. f(g(h))

    :param f: The function through which we pull out
    :param A: The type of the term that we pull out
    :return: B f - The new function through which we have pulled out
    """
    typs = []
    simple_type = f.simple_type
    for _ in f.args:
        typs.insert(0, simple_type.input)
        simple_type = simple_type.output
    assert(f.final_type == simple_type)

    final_type = (A >> f.final_type.input) >> (A >> f.final_type.output)
    new_type = final_type
    for typ in typs:
        new_type = typ >> new_type

    return Term(f.name, new_type, final_type, f.args)


def exch_t(term, i):
    typs = []
    t = term.simple_type
    for _ in range(len(term.args)):
        typs.insert(0, t.input)
        t = t.output
    t = typs[i] >> t
    for j, typ in enumerate(typs):
        if j != i:
            t = typ >> t
    return Term(term.name, t, t, [])


def exch(term, i):
    g = exch_t(term, i)
    args = term.args
    for j, arg in enumerate(term.args):
        if j != len(args) - i - 1:
            g = g(arg)
    g = g(args[len(args) - i - 1])
    return g


# %%
def check(f, g, h):
    # don't pull out higher order diagrams
    if isinstance(h.final_type, Func):
        return False
    # only pull out from a higher order box:
    if not is_higher_order(f.final_type):
        return False
    return True


def pull_out(term):
    f = just(term)
    for arg in term.args:
        new_arg = pull_out(arg)

        # Try pulling out the argument
        pulled_out = False
        for i in range(len(new_arg.args)):
            try_arg = exch(new_arg, i)
            g, h = pop(try_arg)

            # Only pull out of higher order boxes and
            # don't pull out higher order diagrams
            if is_higher_order(f.final_type) and \
                    not isinstance(h.final_type, Func):
                f = b_combinator(f, h.final_type)
                f = pull_out(f(g))(h)
                pulled_out = True
                break

        if not pulled_out:
            f = f(new_arg)
    return f


# def pull_out(term):
#     # Only pull out from a higher order box.
#     # Otherwise, only apply pull_out to arguments.
#     if not is_higher_order(term.final_type):
#         f = just(term)
#         for arg in term.args:
#             new_arg = pull_out(arg)
#             f(new_arg)
#         return f
#
#     f = just(term)
#     for arg in term.args:
#         new_arg = pull_out(arg)
#
#         # Try pulling out the argument
#         pulled_out = False
#         for i in range(len(new_arg.args)):
#             try_arg = exch(new_arg, i)
#             g, h = pop(try_arg)
#
#             # Don't pull out higher order diagrams
#             if not isinstance(h.final_type, Func):
#                 f = b_combinator(f, h.final_type)
#                 f = pull_out(f(g))(h)
#                 pulled_out = True
#                 break
#
#         # If argument can't be pulled out, apply it
#         if not pulled_out:
#             f = f(new_arg)
#     return f

def s_expand_t(t):
    from discocirc.closed import Ty
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



def s_expand(term):
    """ expand the simple type s to match the number of n types """
    f = just(term)
    t = s_expand_t(term.simple_type)
    f = Term(term.name, t, t, [])

    for arg in term.args:
        f = f(s_expand(arg))
    return f


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


