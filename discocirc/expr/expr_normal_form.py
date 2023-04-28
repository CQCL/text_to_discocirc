from expr import Expr
from discocirc.helpers.closed import Func
from discocirc.helpers.discocirc_utils import change_expr_typ


def expr_normal_form(expr):
    """
    Given an expr put it into normal form.

    :param expr: The expr to be put into normal form.
    :return: A new expr in normal form based on the input expr.
    """

    if expr.expr_type == "literal":
        return Expr.literal(expr.name, expr.typ, expr.head)
    elif expr.expr_type == "lambda":
        return Expr.lmbda(expr_normal_form(expr.var),
                          expr_normal_form(expr.body), expr.head)
    elif expr.expr_type == "list":
        return Expr.lst([expr_normal_form(e) for e in expr.expr_list],
                        interchange=expr.interchange,
                        head=expr.head)
    elif expr.expr_type == "application":
        body = expr
        state_vars = []
        function_vars = []

        while body.expr_type == "application":
            if isinstance(body.arg.typ, Func):
                function_vars.append(body.arg)
            else:
                state_vars.append(body.arg)
            body = body.fun

        body = expr_normal_form(body)

        new_type = body.typ
        for _ in function_vars + state_vars:
            new_type = new_type.output

        for var in state_vars + function_vars:
            new_type = var.typ >> new_type

        new_expr = change_expr_typ(body, new_type)

        for var in reversed(state_vars + function_vars):
            new_expr = new_expr(var)

        return new_expr