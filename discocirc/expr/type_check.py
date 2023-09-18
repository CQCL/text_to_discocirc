from discocirc.helpers.closed import Func, Ty, types_match_modulo_curry, uncurry_types


def type_check(expr):
    """
    Checks if the expression is well-typed. Returns the type if it is, else reutrns False
    """
    if expr.expr_type == "literal":
        return expr.typ

    elif expr.expr_type == "list":
        expected_type = Ty()
        interchanged_expected_type_input = Ty()
        interchanged_expected_type_output = Ty()
        for e in expr.expr_list:
            element_type = type_check(e)
            if not element_type:
                return False
            expected_type = expected_type @ element_type
            if isinstance(element_type, Func):
                element_type = uncurry_types(element_type, uncurry_everything=True)
                interchanged_expected_type_input @= element_type.input
                interchanged_expected_type_output @= element_type.output
            else:
                interchanged_expected_type_output @= element_type
        interchanged_expected_type = interchanged_expected_type_input >> interchanged_expected_type_output

        if expr.typ == expected_type:
            return expected_type
        elif expr.typ == interchanged_expected_type:
            return interchanged_expected_type
        return False

    elif expr.expr_type == "application":
        type_arg = type_check(expr.arg)
        type_expr = type_check(expr.fun)

        if not type_arg or not type_expr:
            return False

        if expr.typ != type_expr.output or not types_match_modulo_curry(type_expr.input, type_arg):
            return False

        return expr.typ
    elif expr.expr_type == "lambda":
        type_var = type_check(expr.var)
        type_expr = type_check(expr.body)

        if not type_var or not type_expr:
            return False

        if not types_match_modulo_curry(expr.typ, type_var >> type_expr):
            return False

        return expr.typ
