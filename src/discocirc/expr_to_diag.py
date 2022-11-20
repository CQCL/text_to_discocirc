from discopy import monoidal, Diagram
from discopy.monoidal import Layer

from discocirc import closed
from discocirc.closed import Func
from discocirc.frame import Frame


def _literal_to_diag(expr, context):
    if expr in context:
        return monoidal.Id(expr.final_type), [expr]

    output_type = expr.final_type
    inputs = []
    while isinstance(output_type, Func):
        inputs.insert(0, output_type.input)
        output_type = output_type.output

    input_type = monoidal.Ty() if len(inputs) == 0 else monoidal.Ty.tensor(*inputs)

    return monoidal.Box(expr.name, input_type, output_type), inputs

def add_swaps(types, no_swaps, final_pos):
    swaps = monoidal.Id(types)
    for i in reversed(range(1, no_swaps + 1)):
        swap_layer = monoidal.Id(types[:-i - 1 - final_pos]) @ \
               Diagram.swap(monoidal.Ty(types[-i - final_pos]), monoidal.Ty(types[-i - 1 - final_pos]))
        if i + final_pos != 1:
            swap_layer = swap_layer @ monoidal.Id(types[-i + 1 - final_pos:])
        swaps = swap_layer >> swaps
        types = swaps.dom
    return swaps

def _lambda_to_diag(expr, context):
    context.add(expr.var)
    body, inputs = _expr_to_diag(expr.expr, context)

    occurance_counter = 0
    new_inputs = []
    for i, inp in enumerate(reversed(inputs)):
        if inp == expr.var:
            if (i > 0):
                swap_layer = add_swaps(body.dom, i - occurance_counter, occurance_counter)
                body = swap_layer >> body

            occurance_counter += 1
        else:
            new_inputs.insert(0, inp)

    if occurance_counter == 0:
        copy_box_output = monoidal.Ty()
    else:
        copy_box_output = monoidal.Ty.tensor(*[expr.var.final_type for _ in range(occurance_counter)])

    if occurance_counter == 0:
        copy_box = monoidal.Box(f"del: {expr.var}", expr.var.final_type, copy_box_output)
    elif occurance_counter == 1:
        copy_box = monoidal.Id(expr.var.final_type)
    else:
        copy_box = monoidal.Box(f"copy: {expr.var}", expr.var.final_type,
                                copy_box_output)

    body = monoidal.Id(body.dom[:-occurance_counter]) @ copy_box >> body

    return body, new_inputs + [expr.var.final_type]

def get_next_input(inputs):
    for i in range(1, len(inputs) + 1):
        if isinstance(inputs[-i], closed.Ty):
            next_input = inputs[-i]
            index = len(inputs) - i
            break

    return next_input, index

def _application_to_diag(expr, context):
    arg, arg_inputs = _expr_to_diag(expr.arg, context)
    body, body_inputs = _expr_to_diag(expr.expr, context)

    next_input, input_index = get_next_input(body_inputs)

    if not isinstance(next_input, Func):
        new_args = monoidal.Id(body.dom[:input_index - len(arg.cod) + 1]) @ \
                    arg @ \
                    monoidal.Id(body.dom[input_index + 1:])
        new_inputs = body_inputs[:input_index - len(arg.cod) + 1] + arg_inputs + \
                    body_inputs[input_index + 1:]
        return new_args >> body, new_inputs

    else:
        new_dom = body.dom[:input_index - len(arg.cod) + 1] @ \
                    body.dom[input_index + 1:]
        new_inputs = body_inputs[:input_index - len(arg.cod) + 1] + \
                    body_inputs[input_index + 1:]

        # TODO: find the right frame and don't delete rest of diagram
        inputs = monoidal.Id(new_dom)
        wire_delete_idx = input_index
        for left, box, right in body.layers[:-1]:
            if len(left) > wire_delete_idx:
                left = left[:wire_delete_idx] @ left[wire_delete_idx + 1:]
            elif len(left) == wire_delete_idx:
                raise RuntimeError('Something wrong')
            else:
                idx = wire_delete_idx - len(left) - 1
                right = right[:idx] @ right[idx + 1:]
            inputs = inputs >> (monoidal.Id(left) @ box @ monoidal.Id(right))

        if isinstance(body[-1], Frame):
            body[-1].insides.append(arg)
            frame = body[-1]
        else:
            frame = Frame(body[-1], inputs.cod, body.cod, [arg])

        return inputs >> frame, new_inputs



def _list_to_diag(expr, context):
    output, inputs = _expr_to_diag(expr.expr_list[0], context)
    for val in expr.expr_list[1:]:
        diag, diag_inputs = _expr_to_diag(val, context)
        output = output @ diag
        inputs = inputs + diag_inputs

    return output, inputs


def _expr_to_diag(expr, context):
    if expr.expr_type == "literal":
        return _literal_to_diag(expr, context)
    elif expr.expr_type == "lambda":
        return _lambda_to_diag(expr, context)
    elif expr.expr_type == "application":
        return _application_to_diag(expr, context)
    elif expr.expr_type == "list":
        return _list_to_diag(expr, context)
    raise NotImplementedError(expr.expr_type)

def expr_to_diag(expr):
    diag, _ = _expr_to_diag(expr, set())
    return diag