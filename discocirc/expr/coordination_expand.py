from discocirc.expr.expr import Expr
from discocirc.expr.pull_out import is_higher_order
from discocirc.helpers.closed import Ty
from discocirc.helpers.discocirc_utils import apply_at_root, change_expr_typ, count_applications, create_random_variable


def coordination_expand(expr):
    """
    This function expands coordinating conjunctions like 'and', 'or', etc. between nouns
    in the expr into a higher order expression.
    """
    if expr.expr_type == "application":
        original_head = expr.head
        # recurse in from outside, along arg axis
        for n in range(count_applications(expr, branch='arg'), 0, -1):
            nth_arg = expr
            funs = []
            heads = []
            # pluck off funs, going along arg axis
            for _ in range(n):
                funs.append(nth_arg.fun)
                heads.append(nth_arg.head)
                nth_arg = nth_arg.arg
            # TRIGGER for doing the surgery on the tree: 
            # first check if nth_arg is multiheaded n-type nonliteral
            if nth_arg.typ == Ty('n') and nth_arg.head and len(nth_arg.head) > 1 and nth_arg.expr_type != 'literal':
                # also check the overall function of nth_arg is NOT already higher order
                nth_arg_deep_fun = nth_arg
                for _ in range(count_applications(nth_arg)):
                    nth_arg_deep_fun = nth_arg_deep_fun.fun
                if not is_higher_order(nth_arg_deep_fun.typ): 
                    var = create_random_variable(nth_arg.typ)
                    var.head = nth_arg.head
                    body = var
                    # create the 'composition' of all funs
                    for f, head in zip(reversed(funs), reversed(heads)):
                        # f = coordination_expand(f) # think this became unnecessary ?
                        body = f(body)
                        body.head = head
                    composition = Expr.lmbda(var, body)
                    # attach 'composition' at root of nth_arg
                    nth_arg = change_expr_typ(nth_arg, composition.typ >> nth_arg.typ)
                    nth_arg = apply_at_root(nth_arg, composition)
                    expr = change_expr_typ(nth_arg, expr.typ)
                    expr.head = original_head
                    assert expr.arg == coordination_expand(expr.arg)
                    break # something changed - break out of loop and recurse on new expr
        # recursion at the end
        fun = coordination_expand(expr.fun)
        arg = coordination_expand(expr.arg) # expect that this one is redundant if the for loop triggered
        expr = fun(arg)
        expr.head = original_head
    elif expr.expr_type == "lambda":
        body = coordination_expand(expr.body)
        expr = Expr.lmbda(expr.var, body)
    return expr # recursion for other types?
