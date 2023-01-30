from discopy import rigid

from discocirc.diag.frame import Frame
from discocirc.helpers.closed import Func, Ty


def _literal_to_diag(expr, context):
    name = expr.name
    if expr in context:
        name = f"context: {expr.name}: {expr.final_type}"

    output = expr.final_type
    if isinstance(output, Func):
        input = Ty()
        while isinstance(output, Func):
            input = output.input @ input
            output = output.output

        return rigid.Box(name, input, output)
    else:
        return rigid.Box(name, Ty(), output)


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
    if expr.arg.expr_type == "list":
        body = expr_to_diag(expr.expr, context)
        for arg_expr in reversed(expr.arg.expr_list):
            arg = expr_to_diag(arg_expr, context)
            body = _compose(arg, body)
        return body

    arg = expr_to_diag(expr.arg, context)
    body = expr_to_diag(expr.expr, context)
    return _compose(arg, body)


def _compose(arg, body):
    if arg.dom == Ty():
        new_args = rigid.Id(body.dom[:-len(arg.cod)]) @ arg
        assert(arg.cod == body.dom[-len(arg.cod):])
        return new_args >> body

    else:
        new_dom = body.dom[:-1]

        # TODO: this assumes that the thing we apply to is on the last layer (Issue #13)
        inputs = rigid.Id(new_dom)
        for left, box, right in body.layers[:-1]:
            assert(len(right) == 0)
            if box.dom == Ty():
                inputs = inputs @ box
            else:
                inputs = inputs >> (rigid.Id(inputs.cod[:-len(box.dom)]) @ box)

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