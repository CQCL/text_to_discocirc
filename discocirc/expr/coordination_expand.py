from discocirc.expr.expr import Expr
from discocirc.helpers.closed import Ty
from discocirc.helpers.discocirc_utils import apply_at_root, change_expr_typ, count_applications, create_random_variable


def coordination_expand(expr):
    """
    This function expands coordinating conjunctions like 'and', 'or', etc. between nouns
    in the expr into a higher order expression.
    """
    if expr.expr_type == "application":
        original_head = expr.head
        fun = coordination_expand(expr.fun)
        expr = fun(expr.arg)
        expr.head = original_head
        for n in range(count_applications(expr, branch='arg'), 0, -1):
            nth_arg = expr
            funs = []
            heads = []
            for _ in range(n):
                funs.append(nth_arg.fun)
                heads.append(nth_arg.head)
                nth_arg = nth_arg.arg
            if nth_arg.typ == Ty('n') and nth_arg.head and len(nth_arg.head) > 1:
                var = create_random_variable(nth_arg.typ)
                var.head = nth_arg.head
                body = var
                for f, head in zip(reversed(funs), reversed(heads)):
                    f = coordination_expand(f)
                    body = f(body)
                    body.head = head
                composition = Expr.lmbda(var, body)
                nth_arg = change_expr_typ(nth_arg, composition.typ >> nth_arg.typ)
                nth_arg = apply_at_root(nth_arg, composition)
                new_expr = change_expr_typ(nth_arg, expr.typ)
                new_expr.head = original_head
                return coordination_expand(new_expr)
    return expr
