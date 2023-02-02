from discocirc.helpers.closed import Ty


def expr_type_check(expr):
    if expr.expr_type == "literal":
        return expr.typ

    elif expr.expr_type == "list":
        expected_type = Ty()
        for expr in expr.expr_list:
            element_type = expr_type_check(expr)
            if element_type is None:
                return None
            expected_type = expected_type @ element_type

        if expr.typ == expected_type:
            return expected_type
        else:
            return None

    elif expr.expr_type == "application":
        type_arg = expr_type_check(expr.arg)
        type_expr = expr_type_check(expr.expr)

        if type_arg is None or type_expr is None:
            return None

        if expr.typ != type_expr.output or type_expr.input != type_arg:
            return None

        return expr.typ
    elif expr.expr_type == "lambda":
        type_var = expr_type_check(expr.var)
        type_expr = expr_type_check(expr.expr)

        if type_var is None or type_expr is None:
            return None

        if expr.typ != type_var >> type_expr:
            return None

        return expr.typ
