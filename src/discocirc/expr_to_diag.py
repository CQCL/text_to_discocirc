from discopy import rigid, Diagram

from discocirc.closed import Func, Ty, uncurry_types
from discocirc.frame import Frame

def _literal_to_diag(expr, context):
    name = expr.name
    if expr in context:
        name = f"context: {expr.name}: {expr.final_type}"

    uncurried = uncurry_types(expr.final_type)
    if isinstance(uncurried, Func):
        return rigid.Box(name, uncurried.input, uncurried.output)
    else:
        return rigid.Box(name, Ty(), uncurried)

def _lambda_to_diag(expr, context):
    context.add(expr.var)
    body = expr_to_diag(expr.expr, context)
    context.remove(expr.var)

    return Frame(f"lambda: {expr.var.name}: {expr.var.final_type}",
                 body.dom @ expr.var.final_type,
                 body.cod,
                 [body]
            )

def _application_to_diag(expr, context):
    arg = expr_to_diag(expr.arg, context)
    body = expr_to_diag(expr.expr, context)

    if not isinstance(expr.arg.final_type, Func):
        new_args = rigid.Id(body.dom[:-len(arg.cod)]) @ arg
        return new_args >> body

    else:
        new_dom = body.dom[:-1]

        # TODO: this assumes that the thing we apply to is on the last layer
        inputs = rigid.Id(new_dom)
        for left, box, right in body.layers[:-1]:
            assert(len(right) == 0)
            # right = right[:-1]
            inputs = inputs >> (rigid.Id(left[:-1]) @ box)

        if isinstance(body.boxes[-1], Frame):
            frame = Frame(body.boxes[-1], inputs.cod, body.cod,
                          body.boxes[-1].insides + [arg])
        else:
            frame = Frame(body.boxes[-1], inputs.cod, body.cod, [arg])

        return inputs >> frame



def _list_to_diag(expr, context):
    output = rigid.Id(Ty())
    for val in expr.expr_list:
        diag = expr_to_diag(val, context)
        output = output @ diag

    return output


def expr_to_diag(expr, context=None):
    if context is None:
        context = set()

    if expr.expr_type == "literal":
        return _literal_to_diag(expr, context)
    elif expr.expr_type == "lambda":
        return _lambda_to_diag(expr, context)
    elif expr.expr_type == "application":
        return _application_to_diag(expr, context)
    elif expr.expr_type == "list":
        return _list_to_diag(expr, context)
    else:
        raise NotImplementedError(expr.expr_type)