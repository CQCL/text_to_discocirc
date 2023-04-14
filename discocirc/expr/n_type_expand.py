
import random
from discocirc.expr.expr import Expr
from discocirc.expr.expr_uncurry import expr_uncurry
from discocirc.expr.s_type_expand import expand_closed_type
from discocirc.helpers.closed import Func, Ty
from discocirc.helpers.discocirc_utils import change_expr_typ, expr_type_recursion


def n_expand(expr):
    if expr.expr_type == "literal":
        new_type = expand_closed_type(expr.typ, Ty('n'))
        return change_expr_typ(expr, new_type)
    elif expr.expr_type == "application":
        head = expr.head
        if expr.arg.typ == Ty('n') and expr.arg.head:
            arg = n_expand(expr.arg)
            fun = expr.fun
            uncurried_arg = expr_uncurry(arg)
            old_uncurried_arg = expr_uncurry(expr.arg)
            if isinstance(uncurried_arg.typ, Func):
                arg_output_wires = len(uncurried_arg.typ.output)
                old_arg_output_wires = len(old_uncurried_arg.typ.output)
            else:
                arg_output_wires = len(uncurried_arg.typ) # find the number of output nouns of the argument
                old_arg_output_wires = len(old_uncurried_arg.typ) # find the number of output nouns of the argument
            if arg.head and old_arg_output_wires != arg_output_wires:
                wire_index = 0
                for e in expr_uncurry(arg).arg.expr_list:
                    if e.typ == Ty('n'):
                        if e.head == arg.head:
                            break
                        wire_index = wire_index + 1
                x = Expr.literal(f"x_{random.randint(1000,9999)}", typ=Ty('n'))
                id_expr = Expr.lmbda(x, x)
                left_ids = [id_expr] * wire_index
                right_ids = [id_expr] * (arg_output_wires - wire_index - 1)
                fun = Expr.lst(left_ids + [fun] + right_ids)
                expr = fun(arg)
                return n_expand(expr)
        fun = n_expand(expr.fun)
        arg = n_expand(expr.arg)
        if isinstance(arg.typ, Func):
            if arg.typ != fun.typ.input:
                fun = change_expr_typ(fun, arg.typ >> fun.typ.output)
        expr = fun(arg)
        expr.head = head
        return expr
    else:
        return expr_type_recursion(expr, n_expand)
